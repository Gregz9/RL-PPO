import gymnasium as gym
import numpy as np
from minigrid.manual_control import ManualControl
from minigrid.wrappers import ImgObsWrapper


env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="rgb_array")
observation, info = env.reset(seed=42)
# manual_control = ManualControl(env, seed=42)
# manual_control.start()
print(observation["image"].flatten())
exit()
for _ in range(1000): 
    action = env.action_space.sample()
    # action = np.int64(input("Enter action to execute: "))
    # print(type(action))
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    if terminated or truncated: 
        obervation, _ = env.reset()
        # print(info)

env.close()
