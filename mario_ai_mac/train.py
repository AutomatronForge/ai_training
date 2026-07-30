import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env_utils import make_vec_env
import viewer


N_ENVS = 8  # M3 has fewer CPU cores — 8 is a safe default


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ViewerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._tick = 0

    def _on_step(self) -> bool:
        self._tick += 1
        if self._tick % 4 != 0:
            return True
        try:
            obs = self.locals.get("obs_tensor")
            if obs is not None:
                frames = obs[:, -1, :, :].cpu().numpy()
                for idx, frame in enumerate(frames):
                    import cv2
                    viewer.update_frame(idx, cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
        except Exception:
            pass
        return True


def main():
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    viewer.start(n_envs=N_ENVS)

    env = make_vec_env(n_envs=N_ENVS)

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path="models/",
            name_prefix="mario_ppo",
        ),
        ViewerCallback(),
    ]

    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        n_steps=1024,
        batch_size=256,
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


if __name__ == "__main__":
    main()
