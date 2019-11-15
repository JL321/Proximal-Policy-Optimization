import gym
import matplotlib.pyplot as plt
from PPO import PPO, ExperienceBuffer
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description = 'Specify run configurations for PPO implementation')
parser.add_argument('--env', type = str, default = 'Humanoid-v2', help = 'Specify environment to run in -\
                    must be compatible with OpenAI Gym.')
parser.add_argument('--episodes', type = int, default = 50, help ='Number of episodes to run')
parser.add_argument('--localsteps', type = int, default = 1000, help = 'Number of time steps run per episode,\
                    also the size of the experience buffer')
parser.add_argument('--printInt', type = int, default = 10, help = 'Evaluate performance every printInt episodes and save')
parser.add_argument('--batchSize', type = int, default = 16, help = 'Batch size to train under')
parser.add_argument('--savePath', type = str, default = 'model', help = 'Save model on specified directory')
args = parser.parse_args()

tf.reset_default_graph()
env = gym.make(args.env)
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

print("Observation Space: {}".format(obs_space))

def trainingLoop(model, buf, episodes = 1000, timein_epoch = 1000, printInt = 50, reward_scale = 1):
    #Buf representing experience buffer  
    
    done = False
    reward_history = []
    ret_history = []
    global_t = 0
    for i in range(episodes):
        epsReward = 0
        obs = env.reset()
        obs = np.expand_dims(obs, axis = 0)
        t = 0
        epCount = 0
        totalRet = 0

        for t in range(timein_epoch):
            action, log_prob, value = model.predictPolicy(obs)
            #print("A: {} LP: {} STD: {}".format(action, log_prob, std))
            
            new_obs, reward, done, _ = env.step(action)
            reward = reward_scale*reward
            
            new_obs = np.reshape(new_obs, (-1, obs_space))
            buf.store(obs, action, reward, value, log_prob)
            epsReward += reward
            global_t += 1
            obs = new_obs            
            if done or t-1 == timein_epoch:
                if done:
                    epCount += 1
                    ret = buf.finish_traj(reward)
                    totalRet += ret
                    obs, reward, done = env.reset(), 0, False
                    obs = np.expand_dims(obs, axis = 0)
                else:
                    buf.finish_traj(model.predictValue(new_obs))
            #sample = model.return_sample(obs)  
            #print(sample)
        trajectory = buf.get()
        model.trainingStep(trajectory)
        if epCount == 0:
            epCount += 1
        reward_history.append(epsReward)
        ret_history.append(totalRet/epCount)
        done = False
        if (i+1) % printInt == 0:
            print("Average of last {} episodes: {}".format(printInt, np.mean(np.array(reward_history[-printInt:]))))
            print("Average return of last {} episodes: {}".format(printInt, np.mean(np.array(ret_history[-printInt:]))))
            print("Episode Reward on Episode {}: {}".format(i, epsReward))
            print("Return on Episode {}: {}".format(i, totalRet/epCount))
            print("Global Time Step: {}".format(global_t)) 
            print("Saving .. ")
            model.save_variables()
    plt.plot(reward_history)
    
if __name__ == '__main__':
   
    if len(np.array(obs_space).shape) != 0:
        print(obs_space.shape)
        state_space = (None, *obs_space)
    else:
        state_space = (None, obs_space)
    
    buf = ExperienceBuffer(state_space[-1], action_space, args.localsteps)
    model = PPO(state_space, action_space, args.batchSize, args.savePath)    
    print("Action space: {}".format(action_space))
    
    trainingLoop(model, buf, args.episodes, args.localsteps, args.printInt)
