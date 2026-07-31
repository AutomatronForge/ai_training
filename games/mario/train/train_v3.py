"""
train_v3.py — Pixel/CNN training on SuperMarioBros-v3 (all 32 levels, random each
episode). CnnPolicy from scratch (or resume own checkpoint); no pretrained.

Feature-parity with the RAM path (train_ram.py): reads config.py, resumes from the
latest own-checkpoint, logs rich mario/* metrics, ported reward shaping + curriculum
stage support. Keeps the pixel-only live viewer frame-push.

Usage:
    python train_v3.py               # v3 (all levels)
"""
import argparse
import multiprocessing
import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils import make_vec_env, init_shared_weights
import viewer
import ollama_coach
import metrics_db


def _load_config():
    """Read tunables from config.py (volume-mounted), fall back to defaults."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "config.py")
    defaults = dict(
        N_ENVS=8, TOTAL_TIMESTEPS=10_000_000, LEARNING_RATE=2.5e-4,
        CLIP_RANGE=0.2, ENT_COEF=0.05, N_STEPS=1024, BATCH_SIZE=1024,
        N_EPOCHS=4, OLLAMA_INTERVAL=5000, CHECKPOINT_FREQ=50_000,
        RESUME=False, START_STAGE=None, NET_ARCH=None, RUN_NAME="",
    )
    try:
        spec = importlib.util.spec_from_file_location("config", path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        for k in defaults:
            if hasattr(cfg, k):
                defaults[k] = getattr(cfg, k)
    except Exception as e:
        print(f"[train_v3] Could not load config.py ({e}) — using defaults")
    defaults["N_ENVS"] = _resolve_n_envs(defaults["N_ENVS"])
    return defaults


def _resolve_n_envs(n_envs):
    """Allow N_ENVS='auto' (or None/0) to auto-size to the box (~1 env per vCPU,
    minus 1 for the main proc, capped at 32). NES stepping is CPU-bound so more
    envs than cores oversubscribes and hurts throughput."""
    if isinstance(n_envs, int) and n_envs > 0:
        return n_envs
    cpus = os.cpu_count() or 8
    resolved = max(1, min(32, cpus - 1))
    print(f"[train_v3] N_ENVS='auto' -> {resolved} (detected {cpus} vCPUs)")
    return resolved


CFG = _load_config()
N_ENVS = CFG["N_ENVS"]


def _find_latest_checkpoint(version):
    """Return (path, steps) of the newest models/mario_<version>_ppo_*_steps.zip,
    or (None, 0) if none exist. Steps are parsed from the filename."""
    import glob, re
    pattern = os.path.join("models", f"mario_{version}_ppo_*_steps.zip")
    best_path, best_steps = None, 0
    for p in glob.glob(pattern):
        m = re.search(r"_(\d+)_steps\.zip$", p)
        if m and int(m.group(1)) >= best_steps:
            best_path, best_steps = p, int(m.group(1))
    return best_path, best_steps


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._tick = 0
        self._rewards = []
        self._deaths = 0
        self._episodes = 0
        self._jump_actions = 0
        self._total_actions = 0
        self._prev_lives = {}
        self._x_positions = []
        # richer metrics
        self._clears = 0                 # total flagpole grabs (deduped)
        self._deathless_clears = 0       # clears in an episode with no death yet
        self._prev_flag = {}             # per-env: was flag already grabbed?
        self._died_this_ep = {}          # per-env: has a death occurred this episode?
        self._got_flag_this_ep = {}      # per-env: flag grabbed this episode?
        self._level_clears = {}          # "world-stage" -> count
        self._last_summary_ts = 0        # step of last printed summary
        self._max_x_ever = 0             # deepest x-position reached
        self._farthest_level = "1-1"     # highest world-stage seen
        # SQLite metrics store (additive; never crashes training if it fails).
        self._ep_max_x = {}              # per-env deepest x this episode
        self._last_gate_wall = time.time()
        self._db = None
        self._run_id = ""
        self._run_name = ""
        try:
            self._run_id, self._run_name = metrics_db.resolve_run_id(
                resume=bool(CFG.get("RESUME")), run_name=CFG.get("RUN_NAME", ""))
            self._db = metrics_db.MetricsDB(metrics_db.DB_PATH)
            self._db.connect()
            print(f"[metrics] SQLite store at {metrics_db.DB_PATH} | run={self._run_name} ({self._run_id})")
        except Exception as e:
            print(f"[metrics] disabled (connect failed): {e}")
            self._db = None

    def get_stats(self):
        total = max(self._total_actions, 1)
        eps = max(self._episodes, 1)
        avg_x = float(np.mean(self._x_positions[-200:])) if self._x_positions else 0
        return {
            "total_steps": self.num_timesteps,
            "avg_x": avg_x,
            "avg_reward": float(np.mean(self._rewards[-200:])) if self._rewards else 0,
            "deaths_per_ep": self._deaths / eps,
            "stuck_pct": max(0.0, 1.0 - avg_x / 500) if avg_x < 500 else 0.0,
            "jump_pct": self._jump_actions / total,
            "weights": ollama_coach.get_weights(),
        }

    def _on_step(self) -> bool:
        self._tick += 1
        self._total_actions += N_ENVS
        actions = self.locals.get("actions", [])
        self._jump_actions += sum(1 for a in actions if a in (2, 3, 4, 5))
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            flag = bool(info.get("flag_get", False))
            lvl = f"{info.get('world', 1)}-{info.get('stage', 1)}"
            # Count each flag grab once (flag_get stays true for several frames)
            if flag and not self._prev_flag.get(env_idx, False):
                self._clears += 1
                self._level_clears[lvl] = self._level_clears.get(lvl, 0) + 1
                deathless = not self._died_this_ep.get(env_idx, False)
                if deathless:
                    self._deathless_clears += 1
                print(f"[!] CLEAR #{self._clears} at step {self.num_timesteps} "
                      f"| level {lvl} | {'DEATHLESS' if deathless else 'had death'} "
                      f"| total deaths so far {self._deaths}")
                self._got_flag_this_ep[env_idx] = True
            self._prev_flag[env_idx] = flag
            if lvl > self._farthest_level:
                self._farthest_level = lvl

            # Track deepest x this episode + global max (used for metrics rows).
            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)
                if x > self._max_x_ever:
                    self._max_x_ever = x
                if x > self._ep_max_x.get(env_idx, 0):
                    self._ep_max_x[env_idx] = x

            # Episode boundary = `dones[env_idx]` (verified: `life` is stuck at 2
            # and never decrements in these single-stage envs, and Monitor's
            # "episode" key doesn't propagate through the wrapper stack — so
            # `dones` from SB3's rollout is the ONLY reliable per-episode signal).
            # An episode that ends WITHOUT a flag grab is a death.
            done = bool(dones[env_idx]) if env_idx < len(dones) else False
            if done:
                if not self._got_flag_this_ep.get(env_idx, False):
                    self._deaths += 1
                    self._died_this_ep[env_idx] = True
                self._episodes += 1
                if "episode" in info:
                    self._rewards.append(info["episode"]["r"])
                # Write the per-episode metrics row BEFORE resetting the flags,
                # so cleared/deathless reflect the episode that just ended.
                if self._db is not None:
                    try:
                        cleared = self._got_flag_this_ep.get(env_idx, False)
                        self._db.insert_episode({
                            "run_id": self._run_id,
                            "run_name": self._run_name,
                            "ts": time.time(),
                            "timestep": self.num_timesteps,
                            "env_idx": env_idx,
                            "level": lvl,
                            "outcome": "clear" if cleared else "death",
                            "deathless": int(cleared and not self._died_this_ep.get(env_idx, False)),
                            "max_x": int(self._ep_max_x.get(env_idx, x)),
                            "death_x": int(info.get("x_pos", 0)),
                            "episode_reward": float(info["episode"]["r"]) if "episode" in info else None,
                            "status": info.get("status", "small"),
                            "coins": int(info.get("coins", 0)),
                            "time_left": int(info.get("time", 0)),
                            "kills": int(info.get("ep_kills", 0)),
                            "powerups": int(info.get("ep_powerups", 0)),
                            "oneups": int(info.get("ep_oneups", 0)),
                        })
                    except Exception as e:
                        print(f"[metrics] insert_episode failed: {e}")
                # reset per-episode trackers for the fresh attempt
                self._died_this_ep[env_idx] = False
                self._got_flag_this_ep[env_idx] = False
                self._prev_flag[env_idx] = False
                self._ep_max_x[env_idx] = 0

        # Live viewer: if color is on, env 0 put its raw RGB frame in info["rgb"];
        # push that. Otherwise push the grayscale observation (fast default).
        if self._tick % 4 == 0:
            try:
                infos = self.locals.get("infos", [])
                if infos and "rgb" in infos[0]:
                    frame = np.asarray(infos[0]["rgb"])
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    viewer.update_frame(0, frame)
                else:
                    obs = self.locals.get("obs_tensor")
                    if obs is not None:
                        import cv2
                        frame = obs[0, -1, :, :].cpu().numpy()
                        if frame.max() <= 1.0:
                            frame = (frame * 255).clip(0, 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        viewer.update_frame(0, cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
            except Exception:
                pass

        # Periodic summary + TensorBoard mario/* scalars every ~50k timesteps
        if self.num_timesteps - self._last_summary_ts >= 50_000:
            prev_ts = self._last_summary_ts
            self._last_summary_ts = self.num_timesteps
            dpc = (self._deaths / self._clears) if self._clears else float("nan")
            eps = max(self._episodes, 1)
            levels = ", ".join(f"{k}:{v}" for k, v in sorted(self._level_clears.items()))
            print(f"[STATS] step={self.num_timesteps} clears={self._clears} "
                  f"deathless_clears={self._deathless_clears} "
                  f"deaths={self._deaths} deaths/clear={dpc:.1f} "
                  f"episodes={self._episodes} farthest={self._farthest_level} "
                  f"levels_cleared[{levels}]")

            log = self.logger
            log.record("mario/clears_total", self._clears)
            log.record("mario/deathless_clears_total", self._deathless_clears)
            log.record("mario/deathless_clear_rate",
                       (self._deathless_clears / self._clears) if self._clears else 0.0)
            log.record("mario/deaths_total", self._deaths)
            log.record("mario/deaths_per_clear",
                       (self._deaths / self._clears) if self._clears else 0.0)
            log.record("mario/deaths_per_episode", self._deaths / eps)
            log.record("mario/episodes_total", self._episodes)
            log.record("mario/clear_rate_per_episode", self._clears / eps)
            log.record("mario/max_x_reached", self._max_x_ever)
            log.record("mario/jump_pct",
                       self._jump_actions / max(self._total_actions, 1))
            for k, v in self._level_clears.items():
                log.record(f"mario/clears_level/{k}", v)
            log.dump(self.num_timesteps)

            # Mirror to the SQLite store: one aggregate snapshot + one coach row.
            # This is the natural per-gate commit point (~once every 50k steps).
            if self._db is not None:
                try:
                    now = time.time()
                    dt = max(now - self._last_gate_wall, 1e-6)
                    fps = (self.num_timesteps - prev_ts) / dt
                    self._db.insert_snapshot({
                        "run_id": self._run_id, "run_name": self._run_name,
                        "ts": now, "timestep": self.num_timesteps,
                        "clears_total": self._clears,
                        "deaths_total": self._deaths,
                        "deathless_clears": self._deathless_clears,
                        "episodes_total": self._episodes,
                        "clear_pct": self._clears / eps,
                        "deathless_rate": (self._deathless_clears / self._clears) if self._clears else 0.0,
                        "deaths_per_clear": (self._deaths / self._clears) if self._clears else 0.0,
                        "max_x_reached": int(self._max_x_ever),
                        "jump_pct": self._jump_actions / max(self._total_actions, 1),
                        "fps": fps,
                    })
                    w = ollama_coach.get_weights()
                    self._db.insert_coach({
                        "run_id": self._run_id, "run_name": self._run_name,
                        "ts": now, "timestep": self.num_timesteps,
                        "progress_bonus": w.get("progress_bonus"),
                        "velocity_bonus": w.get("velocity_bonus"),
                        "stuck_penalty": w.get("stuck_penalty"),
                        "jump_bonus": w.get("jump_bonus"),
                        "stuck_threshold": w.get("stuck_threshold"),
                    })
                    self._last_gate_wall = now
                except Exception as e:
                    print(f"[metrics] snapshot/coach failed: {e}")
        return True


def main(version="v3"):
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | pixel/CNN | version: {version}")

    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)
    shared_weights["_color"] = 0  # viewer color toggle (0=grayscale/fast, 1=color)

    viewer.start(n_envs=N_ENVS, shared_weights=shared_weights)
    stats_cb = StatsCallback()
    ollama_coach.start(stats_fn=stats_cb.get_stats, shared_weights=shared_weights,
                       interval=CFG["OLLAMA_INTERVAL"])

    env = make_vec_env(n_envs=N_ENVS, version=version, stage=CFG.get("START_STAGE"))

    callbacks = [
        CheckpointCallback(
            save_freq=max(CFG["CHECKPOINT_FREQ"] // N_ENVS, 1),
            save_path="models/",
            name_prefix=f"mario_{version}_ppo",
        ),
        stats_cb,
    ]

    resume_path, resume_steps = (None, 0)
    if CFG.get("RESUME"):
        resume_path, resume_steps = _find_latest_checkpoint(version)

    if resume_path:
        print(f"Resuming from {resume_path} (@{resume_steps} steps) with N_ENVS={N_ENVS}")
        model = PPO.load(resume_path, env=env, device=device,
                         tensorboard_log="./tensorboard/")
        remaining = max(CFG["TOTAL_TIMESTEPS"] - resume_steps, 0)
    else:
        policy_kwargs = None
        if CFG.get("NET_ARCH"):
            # For CnnPolicy, net_arch sizes the MLP head after the CNN extractor.
            policy_kwargs = dict(net_arch=list(CFG["NET_ARCH"]))
        model = PPO(
            "CnnPolicy", env, device=device,
            n_steps=CFG["N_STEPS"], batch_size=CFG["BATCH_SIZE"],
            n_epochs=CFG["N_EPOCHS"], learning_rate=CFG["LEARNING_RATE"],
            clip_range=CFG["CLIP_RANGE"], ent_coef=CFG["ENT_COEF"],
            policy_kwargs=policy_kwargs, verbose=1,
            tensorboard_log="./tensorboard/",
        )
        remaining = CFG["TOTAL_TIMESTEPS"]

    print(f"Training started (pixel/CNN, {version}).")
    print(f"  N_ENVS={N_ENVS} | TOTAL_TIMESTEPS={CFG['TOTAL_TIMESTEPS']}"
          f"{f' | resuming, {remaining} remaining' if resume_path else ''}")
    print("  Live viewer:  http://localhost:8080")
    print("  TensorBoard:  http://localhost:6006\n")

    model.learn(total_timesteps=remaining, callback=callbacks,
                reset_num_timesteps=not resume_path)
    model.save(f"models/mario_{version}_final")
    print(f"\nDone. Model saved to models/mario_{version}_final.zip")
    env.close()
    if getattr(stats_cb, "_db", None) is not None:
        try:
            stats_cb._db.close()
        except Exception:
            pass
    manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3")
    args = parser.parse_args()
    main(version=args.version)
