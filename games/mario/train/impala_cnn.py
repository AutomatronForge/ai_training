"""IMPALA-CNN features extractor for SB3 PPO — a deep residual ConvNet with far
more capacity than the default 3-layer NatureCNN, for holding many Mario levels
in one policy (the small net hit a capacity ceiling and forgot levels).

Architecture (Espeholt et al. 2018 / Procgen IMPALA-CNN):
  3 sequential conv blocks with channel depths [16, 32, 32]; each block =
  conv(3x3) -> maxpool(3x3, stride2) -> 2 residual blocks. Each residual block =
  ReLU -> conv(3x3) -> ReLU -> conv(3x3), added to its input (skip). Then
  ReLU -> flatten -> Linear -> ReLU to `features_dim`.

Input: (batch, 4, 84, 84). SB3 normalizes uint8 obs to [0,1] before this runs.
"""
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.relu(x)
        y = self.conv0(y)
        y = torch.relu(y)
        y = self.conv1(y)
        return x + y


class _ImpalaBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = _ResidualBlock(out_ch)
        self.res1 = _ResidualBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class IMPALACNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256,
                 channels=(16, 32, 32)):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]  # 4 stacked frames
        blocks = []
        c_in = n_input
        for c_out in channels:
            blocks.append(_ImpalaBlock(c_in, c_out))
            c_in = c_out
        self.conv = nn.Sequential(*blocks)
        # infer flattened size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, *observation_space.shape)
            n_flat = self.conv(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Linear(n_flat, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)
        x = torch.relu(x)
        x = x.reshape(x.shape[0], -1)
        return torch.relu(self.fc(x))
