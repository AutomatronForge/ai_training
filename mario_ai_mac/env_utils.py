import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import gymnasium
import numpy as np
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


class SkipFrame(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleResize(gymnasium.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)
        import cv2
        self._cv2 = cv2
        self.shape = (shape, shape)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(shape, shape, 1), dtype=np.uint8
        )

    def observation(self, obs):
        obs = self._cv2.cvtColor(obs, self._cv2.COLOR_RGB2GRAY)
        obs = self._cv2.resize(obs, self.shape, interpolation=self._cv2.INTER_AREA)
        return obs[:, :, None]


class MarioReward(gymnasium.Wrapper):
    """Shape rewards to encourage momentum-based jumping over obstacles."""
    def __init__(self, env):
        super().__init__(env)
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0

    def reset(self, **kwargs):
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = info.get("x_pos", 0)
        dx = x - self._prev_x

        # Bonus for reaching new max x (overall progress)
        if x > self._max_x:
            reward += (x - self._max_x) * 0.1
            self._max_x = x

        # Small reward for rightward velocity — going left to gain momentum is ok
        if dx > 0:
            reward += dx * 0.05
        # No penalty for going left — allows backing up for momentum

        # Stuck = not moving at all (neither left nor right)
        if dx == 0:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        # Only penalize complete standstill, not leftward movement
        if self._stuck_steps > 90:
            reward -= 0.5

        # Bonus for jump actions (A button: actions 2,3,4,5 in SIMPLE_MOVEMENT)
        if action in (2, 3, 4, 5):
            reward += 0.05

        self._prev_x = x
        return obs, reward, terminated, truncated, info


def make_mario_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env)
    env = SkipFrame(env, skip=2)
    env = MarioReward(env)
    env = GrayScaleResize(env, shape=84)
    return env


def make_vec_env(n_envs=8):
    env = SubprocVecEnv([make_mario_env] * n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env
