import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def ffNetwork(x, action_dim, name = None):
    
    with tf.variable_scope(name):
        layerSizes = [64, 64]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.tanh)
        z = tf.contrib.layers.fully_connected(z, action_dim, activation_fn = None)
    return z

class PPO:
    
    #PPO- currently designed for a batch size of 1
    def __init__(self, observation_dim, action_dim, batch_size):
        
        self.sess = tf.Session()
        
        eps = 0.1
        self.x = tf.placeholder(dtype = tf.float32, shape = observation_dim)
        self.epsRewards = tf.placeholder(dtype = tf.float32, shape = None)        
        self.adv = tf.placeholder(dtype = tf.float32, shape = ())
        self.batch_size = batch_size
        
        policyOut = ffNetwork(self.x, action_dim*2, name = 'CurrentPolicyNetwork')
        self.sampleOut = policyOut
        policyOld = ffNetwork(self.x, action_dim*2, name = 'OldPolicyNetwork')

        print("Test: {}, {}".format(policyOut.get_shape()[0], policyOut.get_shape()))

        print("Second test: {}, {}".format(policyOut[:, action_dim:].shape, policyOut.shape))
        meanCurrent = policyOut[:, :action_dim]
        stdCurrent = policyOut[:, action_dim:]
        meanOld = policyOld[:, :action_dim]
        stdOld = policyOld[:, action_dim:]

        print("Shapes: {}, {}".format(meanCurrent.shape, stdCurrent.shape))
        self.policyDist = tfp.distributions.MultivariateNormalDiag(meanCurrent, stdCurrent)
        self.policyDistOld = tfp.distributions.MultivariateNormalDiag(meanOld, stdOld)
    
        self.policyOut = self.policyDist.sample()
        self.policyOld = self.policyDistOld.sample()

        print("Policy Sample Shape: {}".format(self.policyOut.shape))
        self.valueOut = ffNetwork(self.x, 1, name = 'ValueNetwork')
        
        bottomClip = (1-eps)*self.adv
        topClip = (1+eps)*self.adv
        print(self.policyDist.log_prob(self.policyOut).shape)
        policyRatio = self.policyDist.log_prob(self.policyOut)/self.policyDistOld.log_prob(self.policyOut)
        clipped_objective = tf.reduce_mean(tf.clip_by_value(policyRatio*self.adv, bottomClip, topClip))       

        self.valueObjective = tf.reduce_mean(tf.square(self.valueOut - self.epsRewards))
        
        policyParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        valueParam = [v for v in tf.trainable_variables() if 'ValueNetwork' in v.name]   
        self.trainPolicy = tf.train.AdamOptimizer(1e-3).minimize(-clipped_objective, var_list = policyParam) #Take the negativie objective to perform gradient ascent
        self.trainValue = tf.train.AdamOptimizer(1e-3).minimize(self.valueObjective, var_list = valueParam)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print("Initialized Model")

    def predictPolicy(self, obs):
        return self.sess.run(self.policyOut, feed_dict = {self.x: obs}), self.policyDist.log_prob(self.policyOut)
    
    def _predictValue(self, obs):
        return self.sess.run(self.valueOut, feed_dict = {self.x: obs})
    
    def computeAdvantage(self, rewards, states, discount = 0.99, lmbda = 0.95):
       
        #State shape - t_step x batch x dim
        advList = np.zeros(states.shape[0]+1)
        for i in reversed(range(states.shape[0])):
            delta = rewards[i] + discount*self._predictValue(states[i]) - self._predictValue(states[i-1])
            advList[i] = delta + discount*lmbda*advList[i+1]
        return advList[:-1]
    
    def trainingStep(self, rewards, obs, gamma = 0.99, mini_batch = 16, epochs = 4):
        
        obs = np.array(obs)
        rewards = np.array(rewards)
        
        #Takes in an input of an episode trajectory
        adv = self.computeAdvantage(rewards, obs)
        returnSet = np.zeros(obs.shape[0]+1)
        #Returns a GAE estimate at every observation step
        
        for i, r in enumerate(reversed(rewards)):
            returnSet[i] = r + gamma*returnSet[i+1]
        returnSet = returnSet[:-1]  #Remove the first reward
        
        #Substituting returns for rewards
        
        obs = np.squeeze(obs)
        #Adv is a one dimensional list
        
        rdIdx = np.arange(obs.shape[0])
        for _ in range(epochs):
            cIdx = 0
            endIdx = mini_batch
            currentPolicy = self.getParam() #Current Policy prior to eval
            while endIdx < obs.shape[0]:
                batchIdx = rdIdx[cIdx: endIdx]
                #values per objective would need to be computed separately, and averaged outside of the main training graph (b, t_step, dim) doesn't work
                self.sess.run(self.trainPolicy, feed_dict = {self.x: obs[batchIdx], self.adv: np.mean(adv[batchIdx])})
                self.sess.run(self.trainValue, feed_dict = {self.x: obs[batchIdx], self.epsRewards: rewards[batchIdx]})
                cIdx += mini_batch
                endIdx += mini_batch
                self.updateOldPolicy(currentPolicy)
                
            batchIdx= rdIdx[cIdx:]
            self.sess.run(self.trainPolicy, feed_dict = {self.x: obs[batchIdx], self.adv: np.mean(adv[batchIdx])})
            self.sess.run(self.trainValue, feed_dict = {self.x: obs[batchIdx], self.epsRewards: rewards[batchIdx]})
            self.updateOldPolicy(currentPolicy)
            np.random.shuffle(rdIdx)
            
    def getParam(self):
        cParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        cParam = sorted(cParam, key = lambda v: v.name)
        return cParam
        
    def updateOldPolicy(self, cParam):
        
        oldParam = [v for v in tf.trainable_variables() if 'OldPolicyNetwork' in v.name]
        oldParam = sorted(oldParam, key = lambda v: v.name)
        assignOps = []
        
        for cP, oP in zip(cParam, oldParam):
            assignOp = tf.assign(oP, cP)
            assignOps.append(assignOp)
        
        self.sess.run(assignOps)
    
    def save_variables(self, path = 'models'):
        self.saver.save(self.sess, path)
    
    def return_sample(self, obs):
        return self.sess.run(self.sampleOut, feed_dict = {self.x: obs})
        
        
