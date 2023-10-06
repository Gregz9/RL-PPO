import numpy as np
import gymnasium as gym

env = gym.make("MiniGrid-UnlockPickup-v0", render_mode="rgb_array")


print(env.observation_space["image"].shape)

tensor = np.zeros((256, 4, 147))
print(tensor.reshape(-1, 147).shape)

print(256*147)

print(np.random.choice(10, 11, replace=False))
