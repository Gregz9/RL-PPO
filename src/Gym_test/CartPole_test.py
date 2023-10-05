import argparse 
import os 
from distutils.util import strtobool 
import time 
import random 
import gymnasium as gym 

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
observation = env.reset()
episodic_return = 0 
for _ in range(1000): 
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated: 
        observation, info = env.reset()
        # print(observation)
        # print(info.items())
        # for item, value in info.items():
            

env.close()
