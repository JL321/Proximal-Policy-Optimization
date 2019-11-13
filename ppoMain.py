import gym
import matplotlib.pyplot as plt
from PPO import PPO
import numpy as np
import tensorflow as tf
import timeit

tf.reset_default_graph()
env = gym.make('HalfCheetah-v2')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

print("Observation Space: {}".format(obs_space))

def trainingLoop(model, episodes = 1000, reward_scale = 1):
     
    done = False
    reward_history = []
    global_t = 0
    for i in range(episodes):
        epsReward = 0
        state_trajectory = []
        reward_trajectory = []
        action_trajectory = []
        logProb = []
        obs = env.reset()
        obs = np.expand_dims(obs, axis = 0)
        t = 0
        
        while not done:
            
            action, log_prob = model.predictPolicy(obs)
            #print("A: {} LP: {} STD: {}".format(action, log_prob, std))
            
            new_obs, reward, done, _ = env.step(action)
            reward = reward_scale*reward
            new_obs = np.reshape(new_obs, (-1, obs_space))
            state_trajectory.append(obs) #We forego np.squeeze for batch size 1 operations
            reward_trajectory.append(reward)
            action_trajectory.append(np.squeeze(action))
            logProb.append(log_prob)
            obs = new_obs 
            epsReward += reward
            t += 1
            global_t += 1
            #sample = model.return_sample(obs)  
            #print(sample)
        trajectory = (reward_trajectory, state_trajectory, action_trajectory, logProb)
        model.trainingStep(trajectory)
        reward_history.append(epsReward)
        done = False
        if (i+1) % 10 == 0:
            print("Last Action: {}".format(action))
            print("Average of last 10 episodes: {}".format(np.mean(np.array(reward_history[-10:]))))
            print("Episode Reward on Episode {}: {}".format(i, epsReward))
            print("Global Time Step: {}".format(global_t)) 

    plt.plot(reward_history)
    
if __name__ == '__main__':
   
    if len(np.array(obs_space).shape) != 0:
        print(obs_space.shape)
        state_space = (None, *obs_space)
    else:
        state_space = (None, obs_space)
    
    model = PPO(state_space, action_space, 16)    
    print("Action space: {}".format(action_space))
    
    trainingLoop(model)
