"""
train_v0.py — Fine-tune from tsilva pretrained checkpoint on World 1-1.
Loads a skip=4 pretrained model and continues training with skip=2 for finer control.
"""
import multiprocessing
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils import make_vec_env, init_shared_weights
import viewer
import ollama_coach
from download_pretrained import download

N_ENVS = 20
LEVEL = "1-1"
PRETRAINED_PATH = f"pretrained/mario_{LEVEL.replace('-', '_')}_pretrained.zip"


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
        self._x_positions = []
        self._rewards = []
        self._deaths = 0
        self._episodes = 0
        self._jump_actions = 0
        self._total_actions = 0
        self._prev_lives = {}

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
            if "episode" in info:
                self._episodes += 1
                self._rewards.append(info["episode"]["r"])
            current_life = info.get("life", 3)
            if current_life < self._prev_lives.get(env_idx, 3):
                self._deaths += 1
            self._prev_lives[env_idx] = current_life
            if info.get("flag_get", False):
                print(f"[!] Mario cleared 1-1 at step {self.num_timesteps}!")
            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)
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
        return True


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Download pretrained checkpoint if not present
    if not os.path.exists(PRETRAINED_PATH):
        download(LEVEL)

    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)

    viewer.start(n_envs=N_ENVS)
    stats_cb = StatsCallback()
    ollama_coach.start(stats_fn=stats_cb.get_stats, shared_weights=shared_weights, interval=5000)

    env = make_vec_env(n_envs=N_ENVS)

    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading pretrained checkpoint: {PRETRAINED_PATH}")
        model = PPO.load(
            PRETRAINED_PATH, env=env, device=device,
            clip_range=0.1,   # tighter clip for fine-tuning
            ent_coef=0.02,    # less entropy — model already knows basics
            learning_rate=1e-4,  # lower lr for fine-tuning
            tensorboard_log="./tensorboard/",
        )
        print("Fine-tuning from pretrained checkpoint.")
    else:
        print("No pretrained checkpoint found — training from scratch.")
        model = PPO(
            "CnnPolicy", env, device=device,
            n_steps=1024, batch_size=1024, n_epochs=4,
            learning_rate=2.5e-4, clip_range=0.2, ent_coef=0.05,
            verbose=1, tensorboard_log="./tensorboard/",
        )

    model.set_env(env)
    model.verbose = 1

    callbacks = [
        CheckpointCallback(save_freq=max(50_000 // N_ENVS, 1), save_path="models/", name_prefix="mario_v0_ppo"),
        stats_cb,
    ]

    print("Training started (v0 — World 1-1).")
    print("  Live viewer:  http://localhost:8080")
    print("  TensorBoard:  http://localhost:6006\n")

    model.learn(total_timesteps=5_000_000, callback=callbacks, reset_num_timesteps=False)
    model.save("models/mario_v0_final")
    print("\nDone. Model saved to models/mario_v0_final.zip")
    env.close()
    manager.shutdown()


if __name__ == "__main__":
    main()
