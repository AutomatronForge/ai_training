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

# All 32 SMB levels (world-stage), for random-stage / all-levels training.
ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]


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
    DEATH_PENALTY   = 25.0    # base penalty for dying (episode ends w/o flag)
    DEATH_PROGRESS_SCALE = 25.0  # extra penalty scaled by fraction of level reached
    # Per-level flagpole x (progress denominator for the death penalty). SMB flag
    # positions vary by level; using 1-1's 3161 everywhere miscalibrated the
    # penalty on the other 31 levels. Known values below; DEFAULT_FLAG_X covers
    # the rest (most levels' flags sit ~2600-3300). progress_frac is clamped to
    # [0,1] so an over/under estimate never breaks the penalty scale.
    DEFAULT_FLAG_X  = 3000
    FLAG_X_BY_LEVEL = {
        "1-1": 3161, "1-2": 2560, "1-3": 2560, "1-4": 2048,
        "2-1": 3161, "2-2": 2560, "2-3": 3584, "2-4": 2048,
        "3-1": 3161, "3-2": 3520, "3-3": 2560, "3-4": 2048,
        "4-1": 3584, "4-2": 3072, "4-3": 2560, "4-4": 2048,
        "5-1": 3161, "5-2": 3400, "5-3": 2560, "5-4": 2048,
        "6-1": 3072, "6-2": 3968, "6-3": 2560, "6-4": 2048,
        "7-1": 3161, "7-2": 2560, "7-3": 3584, "7-4": 2048,
        "8-1": 3968, "8-2": 3584, "8-3": 3584, "8-4": 2048,
    }
    STATUS_RANK     = {"small": 0, "tall": 1, "fireball": 2}
    KILL_SCORES     = {100, 200, 400, 500, 800, 1000, 2000, 4000, 8000}
    FIRE_ACTIONS    = {3, 4}  # SIMPLE_MOVEMENT run/B actions throw fireballs when fiery

    def __init__(self, env, expose_rgb=False):
        super().__init__(env)
        self._expose_rgb = expose_rgb
        self._max_x = 0
        self._prev_x = 0
        self._stuck_steps = 0
        self._prev_score = 0
        self._prev_coins = 0
        self._prev_status = "small"
        self._prev_life = 2
        # per-episode event counters (surfaced via info for the metrics store)
        self._ep_kills = 0
        self._ep_powerups = 0
        self._ep_oneups = 0

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
        self._ep_kills = 0
        self._ep_powerups = 0
        self._ep_oneups = 0
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
                self._ep_kills += 1

        # ── power-ups: getting, using, and losing ─────────────────────────
        status = info.get("status", "small")
        cur_rank = self.STATUS_RANK.get(status, 0)
        prev_rank = self.STATUS_RANK.get(self._prev_status, 0)
        if cur_rank > prev_rank:
            reward += self.POWERUP_BONUS
            self._ep_powerups += 1
        elif cur_rank < prev_rank:
            reward -= self.POWERDOWN_PEN
        if status == "fireball" and action in self.FIRE_ACTIONS:
            reward += self.FIREBALL_USE

        # 1-up: life increases (rare/absent in single-stage envs where life is
        # stuck at 2, but correct for the full-game path). Tracked per episode.
        life_now = info.get("life", 2)
        if life_now > self._prev_life:
            self._ep_oneups += 1

        # ── death penalty (progress-scaled) ───────────────────────────────
        # This ROM has INFINITE lives: `life` is stuck at 2 and never
        # decrements, so the old `life < prev_life` check NEVER fired and death
        # was effectively free. A death here = the episode ends WITHOUT grabbing
        # the flag (Mario is reset to the start). Penalize that, scaled by how
        # much progress was thrown away — dying near the flag costs the most,
        # so the agent learns not to waste a good run.
        died = (terminated or truncated) and not bool(info.get("flag_get", False))
        if died:
            lvl = f"{info.get('world', 1)}-{info.get('stage', 1)}"
            flag_x = self.FLAG_X_BY_LEVEL.get(lvl, self.DEFAULT_FLAG_X)
            progress_frac = min(self._max_x / flag_x, 1.0)
            reward -= self.DEATH_PENALTY + progress_frac * self.DEATH_PROGRESS_SCALE

        # enemy-dodge / jump bonuses (coach-tuned)
        if self._enemy_near(x) and action in (2, 3, 4, 5):
            reward += 0.5  # enemies are a common mid-level blocker
        if not near_pipe and action in (2, 3, 4, 5):
            reward += self._w("jump_bonus", 0.05)

        self._prev_x = x
        self._prev_score = score
        self._prev_coins = coins
        self._prev_status = status
        self._prev_life = life_now
        # Surface per-episode event counters + current coins for the metrics
        # store (read by the callback at the episode boundary).
        info["ep_kills"] = self._ep_kills
        info["ep_powerups"] = self._ep_powerups
        info["ep_oneups"] = self._ep_oneups
        # Full-color viewer (opt-in): env 0 attaches its raw NES frame ONLY when
        # the shared color flag is on (toggled from the viewer). Off by default
        # so there's zero per-step cost during normal training.
        if self._expose_rgb and get_shared_weights().get("_color", 0):
            try:
                info["rgb"] = self.env.unwrapped.screen
            except Exception:
                pass
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


def make_mario_env(version="v3", stage=None, expose_rgb=False):
    # stage like "1-2" -> train directly on that stage (SuperMarioBros-1-2-v0),
    # else the plain versioned env (v0 = start at 1-1 and auto-advance; v3 = random).
    env_id = f"SuperMarioBros-{stage}-{version}" if stage else f"SuperMarioBros-{version}"
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymV21CompatibilityV0(env=env)
    env = SkipFrame(env, skip=2)
    env = MarioReward(env, expose_rgb=expose_rgb)
    env = GrayScaleResize(env, shape=84)
    return env


def make_vec_env(n_envs=8, version="v3", stage=None, random_stages=False):
    import functools
    # Only env 0 can expose its raw RGB frame (and only when the viewer's color
    # toggle is on) — the one env the viewer shows. Keeps the color cost off the
    # other 30 envs entirely.
    # random_stages: spread the 32 levels across the env vector (each env pinned
    # to a different level, round-robin) so every gradient step sees a mixture of
    # levels — the anti-forgetting cure for a generalist across all levels.
    if random_stages:
        stages = [ALL_STAGES[i % len(ALL_STAGES)] for i in range(n_envs)]
    else:
        stages = [stage] * n_envs
    env_fns = [
        functools.partial(make_mario_env, version=version, stage=stages[i],
                          expose_rgb=(i == 0))
        for i in range(n_envs)
    ]
    env = SubprocVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env
