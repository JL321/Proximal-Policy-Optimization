import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import scipy.signal
import os

EPS = 1e-8

def ffNetwork(x, action_dim, name = None):
    
    with tf.variable_scope(name):
        layerSizes = [64, 64]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.tanh)
        mean = tf.contrib.layers.fully_connected(z, action_dim, activation_fn = None)
    return mean

def valueNetwork(x, action_dim, name = None):
    with tf.variable_scope(name):
        layerSizes = [64, 64]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.tanh)
        z = tf.contrib.layers.fully_connected(z, 1, activation_fn = None)
    return z

class ExperienceBuffer():
    
    def __init__(self, observation_dim, action_dim, max_len = 1000):

        #Taken from spinning it up - OpenAI
        self.obs_list = np.zeros((max_len, observation_dim))
        self.action_list = np.zeros((max_len, action_dim))
        self.reward_list = np.zeros(max_len)
        self.return_list = np.zeros(max_len)
        self.adv_list = np.zeros(max_len)
        self.logp_list = np.zeros(max_len)
        self.val_list = np.zeros(max_len)
        self.gamma, self.lam = .99, .97
        self.point, self.max_len = 0, 1000
        self.path_start_idx = 0
       
    def store(self, obs, act, rew, val, logp):
        self.obs_list[self.point] = obs
        self.action_list[self.point] = act
        self.reward_list[self.point] = rew
        self.val_list[self.point] = val
        self.logp_list[self.point] = logp
        self.point += 1

    def _discount(self, x, discount):
        '''
        for i in reversed(range(discountList.shape[0]-1)):
            discountList[i] = discount*discountList[i+1]
        return discountList
        '''
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_traj(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.point)
        rews = np.append(self.reward_list[path_slice], last_val)
        vals = np.append(self.val_list[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_list[path_slice] = self._discount(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.return_list[path_slice] = self._discount(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.point
        return self.return_list[self.path_start_idx]

    def get(self):
        
        self.point, self.path_start_idx = 0, 0
        return [self.obs_list, self.action_list, self.logp_list, self.adv_list, self.logp_list]

class PPO:
    
    def __init__(self, observation_dim, action_dim, batch_size):
        
        self.sess = tf.Session()
        
        eps = 0.2
        self.x = tf.placeholder(dtype = tf.float32, shape = observation_dim)
        self.action = tf.placeholder(dtype = tf.float32, shape = (None, action_dim))
        self.epsRewards = tf.placeholder(dtype = tf.float32, shape = None)        
        self.adv = tf.placeholder(dtype = tf.float32, shape = None)
        self.batch_size = batch_size
        self.old_log_prob = tf.placeholder(dtype = tf.float32, shape = None)

        meanCurrent = ffNetwork(self.x, action_dim, name = 'CurrentPolicyNetwork')
        #meanCurrent = modelOut[:, :action_dim]
        #logstdCurrent = modelOut[:, action_dim:] 
        #print("Confirm shape: {}".format(meanCurrent))

        #stdCurrent = tf.nn.softplus(tf.tile(tf.expand_dims(tf.log(tf.exp(0.35)-1), 0), [action_dim])) 
        #stdCurrent = tf.exp(logstdCurrent)
        #self.outStd = stdCurrent
        stdCurrent = tf.get_variable(name='log_std', initializer=-0.5*np.ones(action_dim, dtype=np.float32))
        std = tf.exp(stdCurrent)

        print("Shapes: {}, {}".format(meanCurrent.shape, stdCurrent.shape))
        #self.policyDist = tfp.distributions.MultivariateNormalDiag(meanCurrent, stdCurrent)
        #self.policyEntropy = self.policyDist.entropy()
        self.policyOut = meanCurrent + tf.random_normal(tf.shape(meanCurrent))*stdCurrent

        print("Policy Sample Shape: {}".format(self.policyOut.shape))
        self.valueOut = valueNetwork(self.x, 1, name = 'ValueNetwork')
        
        bottomClip = (1-eps)
        topClip = (1+eps)
        min_adv = tf.where(self.adv > 0, topClip*self.adv, bottomClip*self.adv) 
        #Log prob in negative terms (log_prob = neg_log_prob)
        self.current_log_prob = self.neg_log_prob(self.action, meanCurrent, stdCurrent)
        self.policy_log_prob = self.neg_log_prob(self.policyOut, meanCurrent, stdCurrent)

        policyRatio = tf.exp(self.current_log_prob - self.old_log_prob)
        clipped_objective = tf.reduce_mean(tf.minimum(min_adv, policyRatio*self.adv))

        self.valueObjective = (1/2)*tf.reduce_mean((self.valueOut - self.epsRewards)**2)
        #self.combinedLoss = -clipped_objective + self.valueObjective
        
        policyParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        valueParam = [v for v in tf.trainable_variables() if 'ValueNetwork' in v.name]   
        self.trainPolicy = tf.train.AdamOptimizer(3e-4).minimize(-clipped_objective, var_list = policyParam) #Take the negativie objective to perform gradient ascent
        self.trainValue = tf.train.AdamOptimizer(1e-3).minimize(self.valueObjective, var_list = valueParam)
        #self.trainModel = tf.train.AdamOptimizer(1e-5).minimize(self.combinedLoss)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print("Initialized Model")

    def predictPolicy(self, obs):
        return self.sess.run([self.policyOut, self.policy_log_prob, self.valueOut], feed_dict = {self.x: obs})
    
    def predictValue(self, obs):
        return self.sess.run(self.valueOut, feed_dict = {self.x: obs})
   
    def compute_entropy(self, probs):
        return tf.reduce_sum(probs*tf.log(probs))

    def neg_log_prob(self, x, mu, log_std):
        #Returns the negative log pdf for a diagonal multivariate gaussian
        pre_sum = 0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
        #return (int(x.get_shape()[-1])/2)*tf.log(2*np.pi) + tf.reduce_sum(tf.log(std), axis = -1) + (0.5)*tf.reduce_sum(tf.square((x-mean)/std), axis = -1) #Axis = -1 to sum across normal dim

    def computeAR(self, rewards, states, values, discount = 0.99, lmbda = 0.95, useGAE = True):
       
        #Computes advantage and return
        #State shape - t_step x batch x dim
        advList = np.zeros(rewards.shape)
        deltas = rewards[:-1] + discount*values[1:] - values[:-1]
        
        advList = self._discount(deltas, lmbda*discount)
        returnList = self._discount(rewards, discount)[:-1]

        return advList,returnList
    
    def _discount(self, x, discount):
        '''
        for i in reversed(range(discountList.shape[0]-1)):
            discountList[i] = discount*discountList[i+1]
        return discountList
        '''
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
        
    def trainingStep(self, buffer_traj, gamma = 0.99, mini_batch = 64, epochs = 10):
        
        obs, actions, logprobs, adv, returnSet = buffer_traj

        #Takes in an input of an episode trajectory
        #adv, returnSet = self.computeAR(rewards, obs, values)
        adv = (adv-np.mean(adv))/(np.std(adv)+1e-8) #Normalize advantage estimate - 1e-8 to prevent dividing by 0
        adv = np.squeeze(adv)
        #Returns a GAE estimate at every observation step

        obs = np.squeeze(obs)
        #Adv is a one dimensional list
        
        rdIdx = np.arange(obs.shape[0])
        for _ in range(epochs):
            cIdx = 0
            endIdx = mini_batch
            #currentPolicy = self.getParam() #Current Policy prior to eval
        
            while endIdx < obs.shape[0]:
                batchIdx = rdIdx[cIdx: endIdx]
                
                self.sess.run(self.trainPolicy, feed_dict = {self.x: obs[batchIdx], self.adv: adv[batchIdx], self.action: actions[batchIdx], self.old_log_prob: logprobs[batchIdx]})
                self.sess.run(self.trainValue, feed_dict = {self.x: obs[batchIdx], self.epsRewards: returnSet[batchIdx]})
   
                cIdx += mini_batch
                endIdx += mini_batch
                #print("Negative log probs: {}".format(sess.run(self.currentLogProb, feed_dict = {self.action: actions[batchIdx]})))
            
            batchIdx= rdIdx[cIdx:]
            #batchIdx = np.arange(obs.shape[0]+1)
            self.sess.run(self.trainPolicy, feed_dict = {self.x: obs[batchIdx], self.adv: adv[batchIdx], self.action: actions[batchIdx], self.old_log_prob: logprobs[batchIdx]})
            self.sess.run(self.trainValue, feed_dict = {self.x: obs[batchIdx], self.epsRewards: returnSet[batchIdx]})
            #self.sess.run(self.trainModel, feed_dict = {self.x: obs[batchIdx], self.adv: adv[batchIdx], self.action: actions[batchIdx], self.old_log_prob: logprobs[batchIdx],\
            #          self.epsRewards: returnSet[batchIdx]})
            #np.random.shuffle(rdIdx)
            
    def getParam(self):
        cParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        cParam = sorted(cParam, key = lambda v: v.name)
        return cParam
    
    '''
    def updateOldPolicy(self, cParam):
        
        oldParam = [v for v in tf.trainable_variables() if 'OldPolicyNetwork' in v.name]
        oldParam = sorted(oldParam, key = lambda v: v.name)
        assignOps = []
        
        for cP, oP in zip(cParam, oldParam):
            assignOp = tf.assign(oP, cP)
            assignOps.append(assignOp)
        
        self.sess.run(assignOps)
    '''

    def save_variables(self, path = 'models'):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.saver.save(self.sess, path)
            
