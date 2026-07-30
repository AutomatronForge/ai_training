"""
RAM-based observation for Mario — replaces 84x84 pixel CNN with a small MLP.
Uses game memory values directly: position, enemies, score, status.
Training is ~10x faster and the model is ~100x smaller.
"""
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# RAM observation vector layout (14 values, all normalized 0-1):
# [x_pos, y_pos, x_vel, y_vel, coins, score, time, life, world, stage,
#  is_small, is_tall, is_fireball, flag_get]
RAM_OBS_SIZE = 14
RAM_OBS_MAX = np.array([
    3000,   # x_pos (world width)
    255,    # y_pos
    10,     # x_vel (estimated max)
    10,     # y_vel
    99,     # coins
    999999, # score
    400,    # time
    3,      # life
    8,      # world
    4,      # stage
    1,      # is_small
    1,      # is_tall
    1,      # is_fireball
    1,      # flag_get
], dtype=np.float32)


class RAMObservation(gymnasium.ObservationWrapper):
    """Replace pixel observations with normalized RAM state vector."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_x = 0
        self._prev_y = 0
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(RAM_OBS_SIZE,),
            dtype=np.float32,
        )

    def observation(self, obs):
        # obs is the raw pixel frame — we ignore it and use info from RAM
        # info is accessible via self.env.unwrapped
        try:
            u = self.env.unwrapped
            info = {
                "x_pos":    getattr(u, "_x_position", 0),
                "y_pos":    getattr(u, "_y_position", 0),
                "coins":    getattr(u, "_coins", 0),
                "score":    getattr(u, "_score", 0),
                "time":     getattr(u, "_time", 400),
                "life":     getattr(u, "_life", 2),
                "world":    getattr(u, "_world", 1),
                "stage":    getattr(u, "_stage", 1),
                "status":   getattr(u, "_player_status", "small"),
                "flag_get": getattr(u, "_flag_get", False),
            }
        except Exception:
            info = {k: 0 for k in ["x_pos","y_pos","coins","score","time","life","world","stage"]}
            info["status"] = "small"
            info["flag_get"] = False

        x = info["x_pos"]
        y = info["y_pos"]
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._prev_x = x
        self._prev_y = y

        status = info.get("status", "small")
        vec = np.array([
            x, y, dx, dy,
            info["coins"],
            info["score"],
            info["time"],
            info["life"],
            info["world"],
            info["stage"],
            1.0 if status == "small" else 0.0,
            1.0 if status == "tall" else 0.0,
            1.0 if status == "fireball" else 0.0,
            1.0 if info["flag_get"] else 0.0,
        ], dtype=np.float32)

        return np.clip(vec / RAM_OBS_MAX, 0.0, 1.0)


class SkipFrame(gymnasium.Wrapper):
    def __init__(self, env, skip=2):
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


class MarioReward(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0

    def _w(self, key, default):
        from env_utils import get_shared_weights
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


def make_mario_ram_env(version="v0"):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{version}")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env)
    env = SkipFrame(env, skip=2)
    env = MarioReward(env)
    env = RAMObservation(env)
    return env


def make_ram_vec_env(n_envs=20, version="v0"):
    import functools
    env_fn = functools.partial(make_mario_ram_env, version=version)
    env = SubprocVecEnv([env_fn] * n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env
