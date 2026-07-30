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
from stable_baselines3.common.monitor import Monitor

# RAM observation vector layout (27 values, all normalized 0-1):
# [x_pos, y_pos, x_vel, y_vel, coins, score, time, life, world, stage,
#  is_small, is_tall, is_fireball, flag_get,
#  near_pipe, very_near_pipe, dist_to_next_pipe,
#  enemy0_dx, enemy0_dy, enemy1_dx, enemy1_dy, enemy2_dx, enemy2_dy,
#  enemy3_dx, enemy3_dy, enemy4_dx, enemy4_dy]
RAM_OBS_SIZE = 27
RAM_OBS_MAX = np.array([
    3000, 255, 10, 10,   # x, y, dx, dy
    99, 999999, 400, 3, 8, 4,  # coins, score, time, life, world, stage
    1, 1, 1, 1,          # status flags, flag_get
    1, 1, 3000,          # near_pipe, very_near_pipe, dist_next_pipe
    # enemy deltas — relative to Mario, clamped to ±256
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
], dtype=np.float32)


class RAMObservation(gymnasium.Wrapper):
    """Replace pixel observations with normalized RAM state vector read from info dict."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_x = 0
        self._prev_y = 0
        self._last_info = {}
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(RAM_OBS_SIZE,),
            dtype=np.float32,
        )

    def _make_obs(self, info):
        PIPE_X = [224, 400, 616, 790, 1000, 1170, 1364, 1668, 1810, 2060, 2430, 2628]
        x = info.get("x_pos", 0)
        y = info.get("y_pos", 0)
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._prev_x = x
        self._prev_y = y
        status = info.get("status", "small")

        # Pipe proximity
        ahead_pipes = [px for px in PIPE_X if px >= x]
        dist_next = (ahead_pipes[0] - x) if ahead_pipes else 3000
        near = 1.0 if dist_next < 48 else 0.0
        very_near = 1.0 if dist_next < 24 else 0.0

        # Enemy positions from RAM (up to 5 enemies)
        enemy_deltas = []
        try:
            ram = self.env.unwrapped.ram
            for i in range(5):
                ex = ram[0x87 + i]
                ey = ram[0xCF + i]
                etype = ram[0x16 + i]
                if etype > 0:  # active enemy
                    enemy_deltas.extend([
                        np.clip(ex - (x % 256), -256, 256),
                        np.clip(ey - y, -256, 256),
                    ])
                else:
                    enemy_deltas.extend([256.0, 256.0])  # no enemy = max distance
        except Exception:
            enemy_deltas = [256.0] * 10

        vec = np.array([
            x, y, dx, dy,
            info.get("coins", 0),
            info.get("score", 0),
            info.get("time", 400),
            info.get("life", 2),
            info.get("world", 1),
            info.get("stage", 1),
            1.0 if status == "small" else 0.0,
            1.0 if status == "tall" else 0.0,
            1.0 if status == "fireball" else 0.0,
            1.0 if info.get("flag_get", False) else 0.0,
            near, very_near, dist_next,
            *enemy_deltas,
        ], dtype=np.float32)
        return np.clip(vec / RAM_OBS_MAX, 0.0, 1.0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset() returns empty info — take one step to get real values
        obs2, _, _, _, info2 = self.env.step(0)
        if info2:
            info = info2
        self._prev_x = info.get("x_pos", 40)
        self._prev_y = info.get("y_pos", 79)
        return self._make_obs(info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._make_obs(info), reward, terminated, truncated, info


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

        # Proximity jump bonus near pipes
        near_pipe = any(abs(x - px) < 32 for px in [224, 400, 616, 790])
        if near_pipe and action in (2, 3, 4, 5):
            reward += 0.5

        # Enemy dodge bonus — reward jumping when enemy is within 48px ahead
        try:
            ram = self.env.unwrapped.ram
            for i in range(5):
                etype = ram[0x16 + i]
                if etype > 0:
                    ex = ram[0x87 + i]
                    dist = ex - (x % 256)
                    if 0 < dist < 48 and action in (2, 3, 4, 5):
                        reward += 0.3
        except Exception:
            pass

        if not near_pipe and action in (2, 3, 4, 5):
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
    env = Monitor(env)
    return env


def make_ram_vec_env(n_envs=20, version="v0"):
    import functools
    env_fn = functools.partial(make_mario_ram_env, version=version)
    env = SubprocVecEnv([env_fn] * n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env
