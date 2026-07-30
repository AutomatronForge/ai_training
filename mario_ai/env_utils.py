import multiprocessing
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

# Shared weights dict — lives in parent process, readable by subprocesses via Manager
_shared_weights = None


def init_shared_weights(manager, default_weights):
    global _shared_weights
    _shared_weights = manager.dict(default_weights)
    return _shared_weights


def get_shared_weights():
    if _shared_weights is not None:
        return dict(_shared_weights)
    from ollama_coach import DEFAULT_WEIGHTS
    return dict(DEFAULT_WEIGHTS)

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
    """Shape rewards using weights from shared memory (updated by Ollama coach)."""
    def __init__(self, env):
        super().__init__(env)
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0

    def _w(self, key, default):
        return get_shared_weights().get(key, default)

    def reset(self, **kwargs):
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = info.get("x_pos", 0)
        dx = x - self._prev_x

        if x > self._max_x:
            reward += (x - self._max_x) * self._w("progress_bonus", 0.1)
            self._max_x = x

        if dx > 0:
            reward += dx * self._w("velocity_bonus", 0.05)

        if dx == 0:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps > self._w("stuck_threshold", 90):
            reward -= self._w("stuck_penalty", 0.5)

        if action in (2, 3, 4, 5):
            reward += self._w("jump_bonus", 0.05)

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
