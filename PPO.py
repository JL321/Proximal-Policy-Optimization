import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def ffNetwork(x, action_dim, name = None):
    
    with tf.variable_scope(name):
        layerSizes = [64]
        z = x
        for layer in layerSizes:
            z = tf.contrib.layers.fully_connected(z, layer, activation_fn = tf.nn.tanh)
        z = tf.contrib.layers.fully_connected(z, action_dim, activation_fn = None)
    return z

class PPO:
    
    #PPO- currently designed for a batch size of 1
    def __init__(self, observation_dim, action_dim):
        
        self.sess = tf.Session()
        
        eps = 0.1
        self.x = tf.placeholder(dtype = tf.float32, shape = observation_dim)
        self.epsRewards = tf.placeholder(dtype = tf.float32, shape = None)        
        self.adv = tf.placeholder(dtype = tf.float32, shape = None)
        
        policyOut = ffNetwork(self.x, action_dim*2, name = 'CurrentPolicyNetwork')
        self.sampleOut = policyOut
        policyOld = ffNetwork(self.x, action_dim*2, name = 'OldPolicyNetwork')

        self.policyOut = tfp.distributions.MultivariateNormalDiag(policyOut[0][:action_dim], policyOut[0][action_dim:]*tf.eye(action_dim)).sample()
        self.policyOld = tfp.distributions.MultivariateNormalDiag(policyOld[0][:action_dim], policyOld[0][action_dim:]*tf.eye(action_dim)).sample()

        self.valueOut = ffNetwork(self.x, 1, name = 'ValueNetwork')
        
        self.advEstimate = (1+tf.sign(self.adv)*eps)*self.adv
        policyRatio = self.policyOut/self.policyOld
        clipped_objective = tf.reduce_mean(tf.minimum(policyRatio*self.adv, self.advEstimate))       

        #Note: Rewards here are rewards to go!!! Not episode rewards - need to change
        self.valueObjective = tf.reduce_mean(tf.square(self.valueOut - self.epsRewards))
        
        policyParam = [v for v in tf.trainable_variables() if 'CurrentPolicyNetwork' in v.name]
        valueParam = [v for v in tf.trainable_variables() if 'ValueNetwork' in v.name]   
        self.trainPolicy = tf.train.AdamOptimizer(1e-3).minimize(-clipped_objective, var_list = policyParam) #Take the negativie objective to perform gradient ascent
        self.trainValue = tf.train.AdamOptimizer(1e-3).minimize(self.valueObjective, var_list = valueParam)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print("Initialized Model")

    def predictPolicy(self, obs):
        return self.sess.run(self.policyOut, feed_dict = {self.x: obs})
    
    def _predictValue(self, obs):
        return self.sess.run(self.valueOut, feed_dict = {self.x: obs})
    
    def computeAdvantage(self, rewards, states, discount = 0.99, lmbda = 0.9):
        
        totalAdv = 0
        for i, (obs, reward) in enumerate(zip(states[:-1], rewards[:-1])):
            delta = reward + discount*self._predictValue(states[i+1]) - self._predictValue(obs)
            totalAdv += np.power(discount*lmbda, i)*delta
        #finalDelta = rewards[-1] - self._predictValue(states[-1])
        #totalAdv += finalDelta
        return totalAdv
    
    def trainingStep(self, rewards, obs, gamma = 0.99):
        adv = self.computeAdvantage(rewards, obs)
        
        trajReturn = 0
        for i, r in enumerate(reversed(rewards)):
            trajReturn += (gamma**i)*r
            rewards[-i] = trajReturn
        #Substituting returns for rewards
        
        obs = np.squeeze(obs)
        #Currently leveraging the benefit of not using batch sizes - under an environment with batch sizes,
        #values per objective would need to be computed separately, and averaged outside of the main training graph (b, t_step, dim) doesn't work
        self.sess.run(self.trainPolicy, feed_dict = {self.x: obs, self.adv: adv})
        self.sess.run(self.trainValue, feed_dict = {self.x: obs, self.epsRewards: rewards})
        
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
        
        
