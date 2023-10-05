import gymnasium as gym
import torch 
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[32,32], vf=[32, 32]))
# env = gym.make("CartPole-v1", render_mode="human")
env = make_vec_env("CartPole-v1", n_envs=1)

# Create the agent
# model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Retrieve the environment
model.learn(total_timesteps=20_000)
model.save("ppo_cartpole")

del model
model = PPO.load("ppo_cartpole", env=env)

# env.close()
# env = gym.make("CartPole-v1", render_mode="human") 

# print(env[0])
# env.close()
obs = env.reset()
exit()
# env1 = gym.make("CartPole-v1", render_mode="human")
# print(obs)
for _ in range(200):
    action = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # print(dones) 
    # if dones: 
        # print(info)
    env.render("human")

env.close()
