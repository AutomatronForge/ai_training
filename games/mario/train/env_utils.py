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


# --- NES Super Mario Bros tilemap (shared with the RAM path) -----------------
# RAM 0x0500..0x069F = a 13-row x 32-col tile grid (2 screens), row-major. A
# nonzero tile is solid; 0 is empty. Row 12 (bottom) is the ground floor: a run
# of zeros there = a PIT. Used by MarioReward's hazard-aware jump nudges. This is
# level-agnostic and independent of the pixel observation (reads raw nes-py RAM).
TILE_BASE = 0x0500
TILE_ROWS = 13
TILE_COLS = 32
FLOOR_ROW = 12


def _read_tiles(ram):
    """Return the 13x32 tile grid, or None if the RAM slice is the wrong size."""
    sl = ram[TILE_BASE:TILE_BASE + TILE_ROWS * TILE_COLS]
    if sl.size != TILE_ROWS * TILE_COLS:
        return None
    return np.asarray(sl).reshape(TILE_ROWS, TILE_COLS)


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
    """Reward shaping (ported from the RAM path): movement (coach-tuned) + score/
    coins/power-ups/kills/death + hazard-aware (wall/pit/enemy) jump nudges.

    Depends only on `info` fields and `self.env.unwrapped.ram` — NOT on the
    observation — so it works identically under pixel (CNN) or RAM observations.
    """
    COIN_BONUS      = 2.0     # per coin collected
    SCORE_BONUS     = 0.01    # per point of in-game score gained
    POWERUP_BONUS   = 15.0    # small->tall or tall->fireball (got a power-up)
    POWERDOWN_PEN   = 10.0    # lost a power-up (hit while big) — softer than a death
    KILL_BONUS      = 5.0     # stomped/killed an enemy (inferred from score jump)
    FIREBALL_USE    = 0.5     # fired while in fireball state (uses the power-up)
    DEATH_PENALTY   = 25.0    # lost a life
    STATUS_RANK     = {"small": 0, "tall": 1, "fireball": 2}
    KILL_SCORES     = {100, 200, 400, 500, 800, 1000, 2000, 4000, 8000}
    FIRE_ACTIONS    = {3, 4}  # SIMPLE_MOVEMENT run/B actions throw fireballs when fiery

    def __init__(self, env):
        super().__init__(env)
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0
        self._prev_score = 0
        self._prev_coins = 0
        self._prev_status = "small"
        self._prev_life = 2

    def _w(self, key, default):
        return get_shared_weights().get(key, default)

    def reset(self, **kwargs):
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0
        self._prev_score = 0
        self._prev_coins = 0
        self._prev_status = "small"
        self._prev_life = 2
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x = info.get("x_pos", 0)
        dx = x - self._prev_x

        # ── movement shaping (coach-tuned) ────────────────────────────────
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

        # Level-agnostic jump nudge: reward jumping when a WALL or a PIT is just
        # ahead (pits are the exact skill the mid-game levels need).
        wall_ahead, pit_ahead = self._hazard_ahead(x, info.get("y_pos", 0))
        near_pipe = wall_ahead
        if wall_ahead and action in (2, 3, 4, 5):
            reward += 0.5
        if pit_ahead and action in (2, 3, 4, 5):
            reward += 0.7

        # ── score / coins ─────────────────────────────────────────────────
        score = info.get("score", 0)
        coins = info.get("coins", 0)
        d_score = score - self._prev_score
        d_coins = coins - self._prev_coins
        if d_coins > 0:
            reward += d_coins * self.COIN_BONUS
        if d_score > 0:
            reward += d_score * self.SCORE_BONUS
            # kill enemy: a score jump matching enemy point values while an enemy
            # is nearby is very likely a stomp/fireball kill
            if d_score in self.KILL_SCORES and self._enemy_near(x):
                reward += self.KILL_BONUS

        # ── power-ups: getting, using, and losing ─────────────────────────
        status = info.get("status", "small")
        cur_rank = self.STATUS_RANK.get(status, 0)
        prev_rank = self.STATUS_RANK.get(self._prev_status, 0)
        if cur_rank > prev_rank:
            reward += self.POWERUP_BONUS
        elif cur_rank < prev_rank:
            reward -= self.POWERDOWN_PEN
        if status == "fireball" and action in self.FIRE_ACTIONS:
            reward += self.FIREBALL_USE

        # ── death penalty (encourages no-death runs) ──────────────────────
        life = info.get("life", 2)
        if life < self._prev_life:
            reward -= self.DEATH_PENALTY

        # enemy-dodge / jump bonuses (coach-tuned)
        if self._enemy_near(x) and action in (2, 3, 4, 5):
            reward += 0.5  # enemies are a common mid-level blocker
        if not near_pipe and action in (2, 3, 4, 5):
            reward += self._w("jump_bonus", 0.05)

        self._prev_x = x
        self._prev_score = score
        self._prev_coins = coins
        self._prev_status = status
        self._prev_life = life
        return obs, reward, terminated, truncated, info

    def _enemy_near(self, x, window=80):
        """True if an active enemy is within `window` px ahead of Mario."""
        try:
            ram = self.env.unwrapped.ram
            for i in range(5):
                if ram[0x16 + i] > 0:
                    dist = ram[0x87 + i] - (x % 256)
                    if 0 < dist < window:
                        return True
        except Exception:
            pass
        return False

    def _hazard_ahead(self, x, y):
        """(wall_ahead, pit_ahead) within ~3 tiles, from the live NES tile grid.

        wall = solid tile at Mario's row; pit = empty floor row (a gap). Used to
        reward jumping at the right moment on any level.
        """
        wall = pit = False
        try:
            grid = _read_tiles(self.env.unwrapped.ram)
            if grid is not None:
                col = (x // 16) % TILE_COLS
                row = int(min(max((y // 16), 0), TILE_ROWS - 1))
                for step_cols in (1, 2, 3):  # within ~3 tiles (48px)
                    c = (col + step_cols) % TILE_COLS
                    if grid[row, c] != 0:
                        wall = True
                    if grid[FLOOR_ROW, c] == 0:
                        pit = True
        except Exception:
            pass
        return wall, pit


def make_mario_env(version="v3", stage=None):
    # stage like "1-2" -> train directly on that stage (SuperMarioBros-1-2-v0),
    # else the plain versioned env (v0 = start at 1-1 and auto-advance; v3 = random).
    env_id = f"SuperMarioBros-{stage}-{version}" if stage else f"SuperMarioBros-{version}"
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env)
    env = SkipFrame(env, skip=2)
    env = MarioReward(env)
    env = GrayScaleResize(env, shape=84)
    return env


def make_vec_env(n_envs=8, version="v3", stage=None):
    import functools
    env_fn = functools.partial(make_mario_env, version=version, stage=stage)
    env = SubprocVecEnv([env_fn] * n_envs)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env
