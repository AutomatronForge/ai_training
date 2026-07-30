"""
train_ram.py — Train Mario using RAM observations instead of pixels.

Benefits vs pixel training:
- ~10x faster training (MlpPolicy vs CnnPolicy)
- ~100x smaller model (50KB vs 21MB)
- No screen capture needed for deployment — read memory directly
- Works identically on any emulator resolution

Usage:
    python train_ram.py              # v0 (World 1-1)
    python train_ram.py --version v3 # all 32 levels
"""
import argparse
import multiprocessing
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils_ram import make_ram_vec_env
import viewer
import ollama_coach
from env_utils import init_shared_weights


def _load_config():
    """Read tunables from config.py (volume-mounted), fall back to defaults."""
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "config.py")
    defaults = dict(
        N_ENVS=20, TOTAL_TIMESTEPS=2_000_000, LEARNING_RATE=3e-4,
        CLIP_RANGE=0.2, ENT_COEF=0.01, N_STEPS=1024, BATCH_SIZE=512,
        N_EPOCHS=8, OLLAMA_INTERVAL=5000, CHECKPOINT_FREQ=50_000,
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
        print(f"[train_ram] Could not load config.py ({e}) — using defaults")
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
    print(f"[train_ram] N_ENVS='auto' -> {resolved} (detected {cpus} vCPUs)")
    return resolved


CFG = _load_config()
N_ENVS = CFG["N_ENVS"]


def _find_latest_checkpoint(version):
    """Return (path, steps) of the newest models/mario_ram_<version>_ppo_*_steps.zip,
    or (None, 0) if none exist. Steps are parsed from the filename."""
    import glob, re
    pattern = os.path.join("models", f"mario_ram_{version}_ppo_*_steps.zip")
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
        self._got_flag_this_ep = {}      # per-env: flag grabbed this episode? (death = ep end w/o flag)
        self._level_clears = {}          # "world-stage" -> count
        self._last_summary_ts = 0        # step of last printed summary
        self._max_x_ever = 0             # deepest x-position reached
        self._farthest_level = "1-1"     # highest world-stage seen

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
        self._total_actions += N_ENVS
        actions = self.locals.get("actions", [])
        self._jump_actions += sum(1 for a in actions if a in (2, 3, 4, 5))
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            flag = bool(info.get("flag_get", False))
            # Count each flag grab once (flag_get stays true for several frames)
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

            # Death detection: on the full game `life` decrements; on single-stage
            # envs (SuperMarioBros-1-2-v0) it does NOT — the episode just terminates
            # without a flag. So count BOTH: a life drop, OR an episode that ended
            # without a flag grab. Keeps deaths/clear meaningful in curriculum mode.
            current_life = info.get("life", 3)
            if current_life < self._prev_lives.get(env_idx, 3):
                self._deaths += 1
                self._died_this_ep[env_idx] = True
            self._prev_lives[env_idx] = current_life

            if "episode" in info:
                # episode just ended: if no flag was grabbed this episode, it died
                if not self._got_flag_this_ep.get(env_idx, False) \
                        and not self._died_this_ep.get(env_idx, False):
                    self._deaths += 1
                    self._died_this_ep[env_idx] = True
                self._episodes += 1
                self._rewards.append(info["episode"]["r"])
                # reset per-episode trackers for the next episode
                self._died_this_ep[env_idx] = False
                self._got_flag_this_ep[env_idx] = False

            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)
                if x > self._max_x_ever:
                    self._max_x_ever = x

        # Periodic summary every ~50k steps: clears, deaths, deaths/clear, levels
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

            # --- TensorBoard scalars (graphed under the "mario/" section) ---
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
            # per-level clear counts, one line per level (e.g. mario/clears_level/1-1)
            for k, v in self._level_clears.items():
                log.record(f"mario/clears_level/{k}", v)
            log.dump(self.num_timesteps)
        return True


class RenderCallback(BaseCallback):
    """Skips rendering for RAM training — no pixel obs available in subprocess."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


def main(version="v0"):
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | RAM obs | version: {version}")

    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)

    viewer.start(n_envs=N_ENVS)
    stats_cb = StatsCallback()
    ollama_coach.start(stats_fn=stats_cb.get_stats, shared_weights=shared_weights,
                       interval=CFG["OLLAMA_INTERVAL"])

    env = make_ram_vec_env(n_envs=N_ENVS, version=version, stage=CFG.get("START_STAGE"))

    callbacks = [
        CheckpointCallback(
            # save_freq counts rollout steps (per-env), not timesteps — divide by N_ENVS
            # so checkpoints land every CHECKPOINT_FREQ *timesteps* as intended.
            save_freq=max(CFG["CHECKPOINT_FREQ"] // N_ENVS, 1),
            save_path="models/",
            name_prefix=f"mario_ram_{version}_ppo",
        ),
        stats_cb,
        RenderCallback(),
    ]

    resume_path, resume_steps = (None, 0)
    if CFG.get("RESUME"):
        resume_path, resume_steps = _find_latest_checkpoint(version)

    if resume_path:
        print(f"Resuming from {resume_path} (@{resume_steps} steps) with N_ENVS={N_ENVS}")
        model = PPO.load(resume_path, env=env, device=device,
                         tensorboard_log="./tensorboard/")
        # Continue toward the same TOTAL_TIMESTEPS target (remaining budget).
        remaining = max(CFG["TOTAL_TIMESTEPS"] - resume_steps, 0)
    else:
        # Policy net size from config (default SB3 [64,64]). Bumped to [256,256] for
        # 1-2: [64,64] mastered 1-1 but plateaued at 1-2's pipe/enemy section across
        # ~4M steps and 4 reward/obs/timing interventions — capacity is the remaining
        # hypothesis. Architecture change => cannot resume old checkpoints (fresh run).
        policy_kwargs = None
        if CFG.get("NET_ARCH"):
            policy_kwargs = dict(net_arch=list(CFG["NET_ARCH"]))
        model = PPO(
            "MlpPolicy",
            env,
            device=device,
            n_steps=CFG["N_STEPS"],
            batch_size=CFG["BATCH_SIZE"],
            n_epochs=CFG["N_EPOCHS"],
            learning_rate=CFG["LEARNING_RATE"],
            clip_range=CFG["CLIP_RANGE"],
            ent_coef=CFG["ENT_COEF"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard/",
        )
        remaining = CFG["TOTAL_TIMESTEPS"]

    print(f"Training started (RAM obs, {version}).")
    print(f"  N_ENVS={N_ENVS} | TOTAL_TIMESTEPS={CFG['TOTAL_TIMESTEPS']}"
          f"{f' | resuming, {remaining} remaining' if resume_path else ''}")
    print("  Live viewer:  http://localhost:8080")
    print("  TensorBoard: http://localhost:6006\n")

    model.learn(total_timesteps=remaining, callback=callbacks,
                reset_num_timesteps=not resume_path)
    model.save(f"models/mario_ram_{version}_final")
    print(f"\nDone. Model saved to models/mario_ram_{version}_final.zip")
    env.close()
    manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0", choices=["v0", "v3"])
    args = parser.parse_args()
    main(args.version)
