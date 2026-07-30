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
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# RAM observation vector layout (29 values, all normalized 0-1):
# [x_pos, y_pos, x_vel, y_vel, coins, score, time, life, world, stage,
#  is_small, is_tall, is_fireball, flag_get,
#  near_wall, very_near_wall, dist_to_next_wall,
#  pit_ahead, dist_to_pit,                                 <-- Option B (pit sense)
#  enemy0_dx, enemy0_dy, enemy1_dx, enemy1_dy, enemy2_dx, enemy2_dy,
#  enemy3_dx, enemy3_dy, enemy4_dx, enemy4_dy]
RAM_OBS_SIZE = 29
RAM_OBS_MAX = np.array([
    3000, 255, 10, 10,   # x, y, dx, dy
    99, 999999, 400, 3, 8, 4,  # coins, score, time, life, world, stage
    1, 1, 1, 1,          # status flags, flag_get
    1, 1, 3000,          # near_wall, very_near_wall, dist_next_wall
    1, 3000,             # pit_ahead, dist_to_pit
    # enemy deltas — relative to Mario, clamped to ±256
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
], dtype=np.float32)


# --- NES Super Mario Bros tilemap (verified empirically in-env) --------------
# RAM 0x0500..0x069F = a 13-row x 32-col tile grid (2 screens), row-major. A
# nonzero tile is solid; 0 is empty. Row 12 (bottom) is the ground floor: a run
# of zeros there = a PIT. Columns scroll with the screen, so Mario's column is
# (screen_x // 16). See test dump: row12 "########........########" = pit at cols 8-15.
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

    def _sense_ahead(self, x, y):
        """Level-agnostic wall AND pit sense, read live from the NES tile grid.

        Scans the columns just ahead of Mario:
          - a solid tile at his own row  -> WALL  (jump over / it blocks him)
          - an EMPTY floor row (row 12)   -> PIT   (a gap he must jump)
        Returns (wall_near, wall_very_near, dist_wall, pit_ahead, dist_pit).
        Any read failure falls back to "all clear" so we never fabricate a hazard.
        """
        dist_wall, dist_pit = 3000.0, 3000.0
        try:
            grid = _read_tiles(self.env.unwrapped.ram)
            if grid is not None:
                col = (x // 16) % TILE_COLS
                row = int(np.clip((y // 16), 0, TILE_ROWS - 1))
                for step in range(1, 7):  # look ~7 tiles (112px) ahead
                    c = (col + step) % TILE_COLS
                    px = float(step * 16)
                    # WALL: solid tile at Mario's row
                    if dist_wall >= 3000.0 and grid[row, c] != 0:
                        dist_wall = px
                    # PIT: floor row empty = gap in the ground ahead
                    if dist_pit >= 3000.0 and grid[FLOOR_ROW, c] == 0:
                        dist_pit = px
                    if dist_wall < 3000.0 and dist_pit < 3000.0:
                        break
        except Exception:
            dist_wall, dist_pit = 3000.0, 3000.0
        wall_near = 1.0 if dist_wall < 48 else 0.0
        wall_very_near = 1.0 if dist_wall < 24 else 0.0
        pit_ahead = 1.0 if dist_pit < 64 else 0.0  # pit within ~4 tiles = act now
        return wall_near, wall_very_near, dist_wall, pit_ahead, dist_pit

    def _make_obs(self, info):
        x = info.get("x_pos", 0)
        y = info.get("y_pos", 0)
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._prev_x = x
        self._prev_y = y
        status = info.get("status", "small")

        # Level-agnostic wall + pit proximity (Option B: real tile grid, incl. gaps).
        near, very_near, dist_next, pit_ahead, dist_pit = self._sense_ahead(x, y)

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
            pit_ahead, dist_pit,
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
    # Fixed shaping weights for score/coins/power-ups/kills/death. Kept as constants
    # (not coach-tuned) so they stay stable; the Ollama coach still tunes movement.
    COIN_BONUS      = 2.0     # per coin collected
    SCORE_BONUS     = 0.01    # per point of in-game score gained
    POWERUP_BONUS   = 15.0    # small->tall or tall->fireball (got a power-up)
    POWERDOWN_PEN   = 10.0    # lost a power-up (hit while big) — softer than a death
    KILL_BONUS      = 5.0     # stomped/killed an enemy (inferred from score jump)
    FIREBALL_USE    = 0.5     # fired while in fireball state (uses the power-up)
    DEATH_PENALTY   = 25.0    # lost a life
    STATUS_RANK     = {"small": 0, "tall": 1, "fireball": 2}
    # Score deltas Mario gets for stomping/killing enemies (points).
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
        from env_utils import get_shared_weights
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
        # ahead (Option B adds pit awareness — pits are what kill it in 1-2+).
        wall_ahead, pit_ahead = self._hazard_ahead(x, info.get("y_pos", 0))
        near_pipe = wall_ahead
        if wall_ahead and action in (2, 3, 4, 5):
            reward += 0.5
        if pit_ahead and action in (2, 3, 4, 5):
            reward += 0.7  # jumping a gap is the exact skill 1-2 needs

        # ── score / coins ─────────────────────────────────────────────────
        score = info.get("score", 0)
        coins = info.get("coins", 0)
        d_score = score - self._prev_score
        d_coins = coins - self._prev_coins
        if d_coins > 0:
            reward += d_coins * self.COIN_BONUS
        if d_score > 0:
            reward += d_score * self.SCORE_BONUS
            # ── kill enemy: a score jump matching enemy point values while an
            #    enemy is nearby is very likely a stomp/fireball kill ──────
            if d_score in self.KILL_SCORES and self._enemy_near(x):
                reward += self.KILL_BONUS

        # ── power-ups: getting, using, and losing ─────────────────────────
        status = info.get("status", "small")
        cur_rank = self.STATUS_RANK.get(status, 0)
        prev_rank = self.STATUS_RANK.get(self._prev_status, 0)
        if cur_rank > prev_rank:
            reward += self.POWERUP_BONUS            # picked up a mushroom/flower
        elif cur_rank < prev_rank:
            reward -= self.POWERDOWN_PEN            # got hit, lost power-up
        if status == "fireball" and action in self.FIRE_ACTIONS:
            reward += self.FIREBALL_USE             # using the fire power-up

        # ── death penalty (encourages no-death runs) ──────────────────────
        life = info.get("life", 2)
        if life < self._prev_life:
            reward -= self.DEATH_PENALTY

        # enemy-dodge / jump bonuses (coach-tuned)
        if self._enemy_near(x) and action in (2, 3, 4, 5):
            reward += 0.3
        if not near_pipe and action in (2, 3, 4, 5):
            reward += self._w("jump_bonus", 0.05)

        self._prev_x = x
        self._prev_score = score
        self._prev_coins = coins
        self._prev_status = status
        self._prev_life = life
        return obs, reward, terminated, truncated, info

    def _enemy_near(self, x):
        """True if an active enemy is within 48px ahead of Mario."""
        try:
            ram = self.env.unwrapped.ram
            for i in range(5):
                if ram[0x16 + i] > 0:
                    dist = ram[0x87 + i] - (x % 256)
                    if 0 < dist < 48:
                        return True
        except Exception:
            pass
        return False

    def _hazard_ahead(self, x, y):
        """(wall_ahead, pit_ahead) within ~2 tiles, from the live NES tile grid.

        wall = solid tile at Mario's row; pit = empty floor row (a gap). Used to
        reward jumping at the right moment on any level (Option B: pit-aware).
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
    return env
