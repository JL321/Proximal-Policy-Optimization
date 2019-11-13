import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def ffNetwork(x, action_dim, name = None):
    
    with tf.variable_scope(name):
        layerSizes = [100, 200]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.tanh)
        mean = tf.contrib.layers.fully_connected(z, action_dim, activation_fn = None)
    return mean

def valueNetwork(x, action_dim, name = None):
    with tf.variable_scope(name):
        layerSizes = [100, 200]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.relu)
        z = tf.contrib.layers.fully_connected(z, 1, activation_fn = None)
    return z

class PPO:
    
    def __init__(self, observation_dim, action_dim, batch_size):
        
        self.sess = tf.Session()
        
        eps = 0.1
        self.x = tf.placeholder(dtype = tf.float32, shape = observation_dim)
        self.action = tf.placeholder(dtype = tf.float32, shape = (None, action_dim))
        self.epsRewards = tf.placeholder(dtype = tf.float32, shape = None)        
        self.adv = tf.placeholder(dtype = tf.float32, shape = None)
        self.batch_size = batch_size
        self.old_log_prob = tf.placeholder(dtype = tf.float32, shape = None)

        meanCurrent = ffNetwork(self.x, action_dim, name = 'CurrentPolicyNetwork')
        
        stdCurrent = tf.nn.softplus(tf.tile(tf.expand_dims(tf.log(tf.exp(0.35)-1), 0), [action_dim])) 
        self.outStd = stdCurrent

        print("Shapes: {}, {}".format(meanCurrent.shape, stdCurrent.shape))
        self.policyDist = tfp.distributions.MultivariateNormalDiag(meanCurrent, stdCurrent)
        self.policyEntropy = self.policyDist.entropy()
        print("Verify: ")
        print(self.policyDist.prob(self.action))
        self.policyOut = self.policyDist.sample()

        print("Policy Sample Shape: {}".format(self.policyOut.shape))
        self.valueOut = valueNetwork(self.x, 1, name = 'ValueNetwork')
        
        bottomClip = (1-eps)
        topClip = (1+eps)
        
        #Log prob in negative terms (log_prob = neg_log_prob)
        self.current_log_prob = self.neg_log_prob(self.action, meanCurrent, stdCurrent)
        self.policy_log_prob = self.neg_log_prob(self.policyOut, meanCurrent, stdCurrent)

        policyRatio = tf.exp(self.current_log_prob - self.old_log_prob)
        clipped_Ratio = tf.clip_by_value(policyRatio, bottomClip, topClip)
        clipped_objective = tf.reduce_mean(tf.minimum(clipped_Ratio*self.adv, policyRatio*self.adv) + 0.01*self.policyEntropy)

        self.valueObjective = (1/2)*tf.reduce_mean((self.valueOut - self.epsRewards)**2)
        #self.combinedLoss = -clipped_objective + self.valueObjective
        '''
        tf.summary.scalar('valueObjective', self.valueObjective)
        tf.summary.scalar('policyObjective', clipped_objective)
        tf.summary.scalar('combined_loss', self.combinedLoss)
        '''
        policyParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        valueParam = [v for v in tf.trainable_variables() if 'ValueNetwork' in v.name]   
        self.trainPolicy = tf.train.AdamOptimizer(1e-4).minimize(-clipped_objective, var_list = policyParam) #Take the negativie objective to perform gradient ascent
        self.trainValue = tf.train.AdamOptimizer(1e-4).minimize(self.valueObjective, var_list = valueParam)
        #self.trainModel = tf.train.AdamOptimizer(1e-5).minimize(self.combinedLoss)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print("Initialized Model")

    def predictPolicy(self, obs):
        return self.sess.run([self.policyOut, self.policy_log_prob], feed_dict = {self.x: obs})
    
    def _predictValue(self, obs):
        return self.sess.run(self.valueOut, feed_dict = {self.x: obs})
   
    def compute_entropy(self, probs):
        return tf.reduce_sum(probs*tf.log(probs))

    def neg_log_prob(self, x, mean, std):
        #Returns the negative log pdf for a diagonal multivariate gaussian
        return (int(x.get_shape()[-1])/2)*tf.log(2*np.pi) + tf.reduce_sum(tf.log(std), axis = -1) + (1/2)*tf.reduce_sum(tf.square((x-mean)/std), axis = -1) #Axis = -1 to sum across normal dim

    def computeAdvantage(self, rewards, states, discount = 0.99, lmbda = 0.95, useGAE = True):
       
        #State shape - t_step x batch x dim
        advList = np.zeros(states.shape[0])
        GAE_term = 0

        advList[-1] = rewards[-1]
        for i in reversed(range(states.shape[0]-1)):
            delta = rewards[i] + discount*self._predictValue(states[i+1]) - self._predictValue(states[i])
            if useGAE:
                GAE_term = discount*lmbda*advList[i+1]
            advList[i] = delta + GAE_term
        return advList
    
    def trainingStep(self, traj, gamma = 0.99, mini_batch = 64, epochs = 80):
        
        rewards, obs, actions, logprobs = traj
        obs = np.array(obs)
        rewards = np.array(rewards)
        actions = np.array(actions)
        logprobs = np.array(logprobs)

        #Takes in an input of an episode trajectory
        adv = self.computeAdvantage(rewards, obs)
        adv = (adv-np.mean(adv))/(np.std(adv)+1e-8) #Normalize advantage estimate - 1e-8 to prevent dividing by 0
        returnSet = np.zeros(obs.shape[0]+1)
        #Returns a GAE estimate at every observation step
    
        for i in reversed(range(rewards.shape[0])):
            returnSet[i] = rewards[i] + gamma*returnSet[i+1]
        returnSet = returnSet[:-1]  #Remove the first return - filler for loop
        
        #Substituting returns for rewards to go
        
        obs = np.squeeze(obs)
        #Adv is a one dimensional list
        
        rdIdx = np.arange(obs.shape[0])
        for _ in range(epochs):
            cIdx = 0
            endIdx = mini_batch
            currentPolicy = self.getParam() #Current Policy prior to eval
            while endIdx < obs.shape[0]:
                batchIdx = rdIdx[cIdx: endIdx]
                #self.sess.run(self.trainModel, feed_dict = {self.x: obs[batchIdx], self.adv: adv[batchIdx], self.action: actions[batchIdx], self.old_log_prob: logprobs[batchIdx],\
                #       self.epsRewards: returnSet[batchIdx]})
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
        self.saver.save(self.sess, path)
            
