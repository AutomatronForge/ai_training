import multiprocessing
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils import make_vec_env, init_shared_weights
import viewer
import ollama_coach


N_ENVS = 20


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class StatsCallback(BaseCallback):
    """Tracks training stats for Ollama coach and sends frames to viewer."""

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

        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if "episode" in info:
                self._episodes += 1
                self._rewards.append(info["episode"]["r"])

            # Count deaths only at the moment life decreases
            current_life = info.get("life", 3)
            prev_life = self._prev_lives.get(env_idx, 3)
            if current_life < prev_life:
                self._deaths += 1
            self._prev_lives[env_idx] = current_life

            if info.get("flag_get", False):
                print(f"[!] Mario cleared the level at step {self.num_timesteps}!")

            x = info.get("x_pos", 0)
            if x > 0:
                self._x_positions.append(x)

        # Send frame to viewer
        if self._tick % 4 == 0:
            try:
                obs = self.locals.get("obs_tensor")
                if obs is not None:
                    frames = obs[:, -1, :, :].cpu().numpy()
                    for idx, frame in enumerate(frames):
                        import cv2
                        # Handle both float [0,1] and uint8 [0,255] obs
                        if frame.max() <= 1.0:
                            frame = (frame * 255).clip(0, 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                        viewer.update_frame(idx, cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), mode="viridis")
            except Exception:
                pass

        return True


def main():
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # Shared weights live in a Manager dict — readable across subprocesses
    manager = multiprocessing.Manager()
    shared_weights = init_shared_weights(manager, ollama_coach.DEFAULT_WEIGHTS)

    viewer.start(n_envs=N_ENVS)

    stats_cb = StatsCallback()

    # Wire Ollama coach to update the shared weights dict directly
    ollama_coach.start(
        stats_fn=stats_cb.get_stats,
        shared_weights=shared_weights,
        interval=5000,
    )

    env = make_vec_env(n_envs=N_ENVS, version="v3")

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/",
            name_prefix="mario_ppo",
        ),
        stats_cb,
    ]

    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        n_steps=1024,
        batch_size=1024,
        n_epochs=4,
        learning_rate=2.5e-4,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    print("Training started.")
    print("  Live viewer:  http://localhost:8080")
    print("  TensorBoard:  http://localhost:6006\n")

    model.learn(
        total_timesteps=10_000_000,
        callback=callbacks,
    )

    model.save("models/mario_ppo_final")
    print("\nTraining complete. Model saved to models/mario_ppo_final.zip")
    env.close()
    manager.shutdown()


if __name__ == "__main__":
    main()
