import numpy as np 
import os
import gymnasium as gym


if __name__ == "__main__": 
    
    def make_env(gym_id: str):
        def created_env(): 
            env = gym.make(gym_id, render_mode="human")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return created_env

    # envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-UnlockPickup-v0")])
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1")])
    observation = envs.reset(seed=42)
    for _ in range(200): 
        action = envs.action_space.sample()
        observation, reward, done, truncated, info = envs.step(action)
        for item, value in info.items(): 
            if isinstance(value[0], dict): 
                if "episode" in value[0].keys(): 
                    print(f"episodic return {value[0]['episode']['r']}")
    # print(help(gym.vector.SyncVectorEnv))
    # print()
   
envs.close()
    
    
    
