import gym
import matplotlib.pyplot as plt
from PPO import PPO, ExperienceBuffer
import numpy as np
import tensorflow as tf
import timeit

tf.reset_default_graph()
env = gym.make('Humanoid-v2')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

print("Observation Space: {}".format(obs_space))

def trainingLoop(model, buf, episodes = 50, timein_epoch = 1000, reward_scale = 1):
    #Buf representing experience buffer  
    
    done = False
    reward_history = []
    global_t = 0
    for i in range(episodes):
        epsReward = 0

        obs = env.reset()
        obs = np.expand_dims(obs, axis = 0)
        t = 0
        
        for t in range(timein_epoch):
            
            action, log_prob, value = model.predictPolicy(obs)
            #print("A: {} LP: {} STD: {}".format(action, log_prob, std))
            
            new_obs, reward, done, _ = env.step(action)
            reward = reward_scale*reward
            
            new_obs = np.reshape(new_obs, (-1, obs_space))
            buf.store(obs, action, reward, value, log_prob)
            
            epsReward += reward
            global_t += 1
            
            if done:
                buf.finish_traj(reward)
                obs, reward, done = env.reset(), 0, False

            #sample = model.return_sample(obs)  
            #print(sample)
        
        trajectory = buf.get()
        model.trainingStep(trajectory)
        reward_history.append(epsReward)
        done = False
        if (i+1) % 10 == 0:
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
    
    buf = ExperienceBuffer(state_space[-1], action_space)
    model = PPO(state_space, action_space, 16)    
    print("Action space: {}".format(action_space))
    
    trainingLoop(model, buf)
