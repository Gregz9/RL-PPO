import torch
import torch.nn as nn
from gymnasium import spaces
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


def init_weightsNbias(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    episode_trigger=lambda t: t % 1 == 0,
                )
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to torche number of unit for torche last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            init_weightsNbias(nn.Conv2d(n_input_channels, 16, (2, 2))),
            nn.ReLU(),
            init_weightsNbias(nn.Conv2d(16, 32, (2, 2))),
            nn.ReLU(),
            init_weightsNbias(nn.Conv2d(32, 64, (2, 2))),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            init_weightsNbias(nn.Linear(n_flatten, features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
# env = make_vec_env("MiniGrid-UnlockPickup-v0", n_envs=1)
env = DummyVecEnv(
    [
        lambda: ImgObsWrapper(
            gym.make("MiniGrid-UnlockPickup-v0", render_mode="rgb_array")
        )
    ]
)

envs = gym.vector.SyncVectorEnv(
    [
        make_env("MiniGrid-UnlockPickup-v0", 1 + i, i, False, "Something")
        for i in range(4)
    ]
)
print(envs.single_observation_space)

env.seed(42)
# env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=3e4)
model.save("ppo_minigrid")
del model
