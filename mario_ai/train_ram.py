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
import ollama_coach
from env_utils import init_shared_weights

N_ENVS = 20


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
            if "episode" in info:
                self._episodes += 1
                self._rewards.append(info["episode"]["r"])
            current_life = info.get("life", 3)
            if current_life < self._prev_lives.get(env_idx, 3):
                self._deaths += 1
            self._prev_lives[env_idx] = current_life
            if info.get("flag_get", False):
                print(f"[!] Mario cleared the level at step {self.num_timesteps}!")
            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)
        return True


def main(version="v0"):
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device} | RAM obs | version: {version}")

    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)

    stats_cb = StatsCallback()
    ollama_coach.start(stats_fn=stats_cb.get_stats, shared_weights=shared_weights, interval=5000)

    env = make_ram_vec_env(n_envs=N_ENVS, version=version)

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/",
            name_prefix=f"mario_ram_{version}_ppo",
        ),
        stats_cb,
    ]

    model = PPO(
        "MlpPolicy",   # small MLP instead of CNN — trains much faster
        env,
        device=device,
        n_steps=1024,
        batch_size=512,
        n_epochs=8,    # more epochs since data is cheap
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    print(f"Training started (RAM obs, {version}).")
    print("  TensorBoard: http://localhost:6006\n")

    model.learn(total_timesteps=2_000_000, callback=callbacks)
    model.save(f"models/mario_ram_{version}_final")
    print(f"\nDone. Model saved to models/mario_ram_{version}_final.zip")
    env.close()
    manager.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v0", choices=["v0", "v3"])
    args = parser.parse_args()
    main(args.version)
