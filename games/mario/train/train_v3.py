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
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils import make_vec_env, init_shared_weights
import viewer
import ollama_coach


def _load_config():
    """Read tunables from config.py (volume-mounted), fall back to defaults."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "config.py")
    defaults = dict(
        N_ENVS=8, TOTAL_TIMESTEPS=10_000_000, LEARNING_RATE=2.5e-4,
        CLIP_RANGE=0.2, ENT_COEF=0.05, N_STEPS=1024, BATCH_SIZE=1024,
        N_EPOCHS=4, OLLAMA_INTERVAL=5000, CHECKPOINT_FREQ=50_000,
        RESUME=False, START_STAGE=None, NET_ARCH=None,
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
    return defaults


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
        self._clears = 0
        self._deathless_clears = 0
        self._prev_flag = {}
        self._died_this_ep = {}
        self._got_flag_this_ep = {}
        self._level_clears = {}
        self._last_summary_ts = 0
        self._max_x_ever = 0
        self._farthest_level = "1-1"

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
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            flag = bool(info.get("flag_get", False))
            lvl = f"{info.get('world', 1)}-{info.get('stage', 1)}"
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

            current_life = info.get("life", 3)
            if current_life < self._prev_lives.get(env_idx, 3):
                self._deaths += 1
                self._died_this_ep[env_idx] = True
            self._prev_lives[env_idx] = current_life

            if "episode" in info:
                if not self._got_flag_this_ep.get(env_idx, False) \
                        and not self._died_this_ep.get(env_idx, False):
                    self._deaths += 1
                    self._died_this_ep[env_idx] = True
                self._episodes += 1
                self._rewards.append(info["episode"]["r"])
                self._died_this_ep[env_idx] = False
                self._got_flag_this_ep[env_idx] = False

            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)
                if x > self._max_x_ever:
                    self._max_x_ever = x

        if self._tick % 4 == 0:
            try:
                obs = self.locals.get("obs_tensor")
                if obs is not None:
                    frames = obs[:, -1, :, :].cpu().numpy()
                    for idx, frame in enumerate(frames):
                        import cv2
                        if frame.max() <= 1.0:
                            frame = (frame * 255).clip(0, 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        viewer.update_frame(idx, cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
            except Exception:
                pass

        if self.num_timesteps - self._last_summary_ts >= 50_000:
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
        return True


def main(version="v3"):
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | pixel/CNN | version: {version}")

    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)

    viewer.start(n_envs=N_ENVS)
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
    manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v3")
    args = parser.parse_args()
    main(version=args.version)
