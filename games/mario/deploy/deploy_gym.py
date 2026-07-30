"""
deploy_gym.py — Run trained RAM model against the same gym-super-mario-bros
emulator used during training. No RetroArch needed.

Usage:
    python deploy_gym.py
    python deploy_gym.py --model models/mario_ram_v0_ppo_1000000_steps.zip
    python deploy_gym.py --version v3  # random levels
    python deploy_gym.py --fps 30      # control speed (0 = max speed)
"""
import argparse
import os
import glob
import time
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def pick_model():
    search_dirs = ["models", "../models", "."]
    found = []
    for d in search_dirs:
        found += glob.glob(os.path.join(d, "*.zip"))
    found = sorted(set(found))

    if not found:
        return input("No models found. Enter full path to model .zip: ").strip().strip('"')

    print("\nAvailable models:")
    for i, f in enumerate(found):
        print(f"  [{i}] {f}")
    print()
    choice = input("Enter number or full path: ").strip()
    if choice.isdigit():
        return found[int(choice)]
    return choice.strip('"')


def main(model_path, version="v0", fps=30, episodes=10):
    import cv2
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
    import gymnasium

    # Note: VecNormalize was used in training but stats aren't saved separately.
    # Use deterministic=False to allow some exploration which helps generalize.
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    print(f"Model loaded. Running on SuperMarioBros-{version}\n")

    # Same obs builder as training
    PIPE_X = [224, 400, 616, 790, 1000, 1170, 1364, 1668, 1810, 2060, 2430, 2628]
    RAM_OBS_MAX = np.array([
        3000, 255, 10, 10,
        99, 999999, 400, 3, 8, 4,
        1, 1, 1, 1,
        1, 1, 3000,
        256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
    ], dtype=np.float32)

    def build_obs(info, prev_x, prev_y, unwrapped):
        x = info.get("x_pos", 0)
        y = info.get("y_pos", 0)
        dx = x - prev_x
        dy = y - prev_y
        status = info.get("status", "small")
        ahead = [px for px in PIPE_X if px >= x]
        dist_next = (ahead[0] - x) if ahead else 3000
        near = 1.0 if dist_next < 48 else 0.0
        very_near = 1.0 if dist_next < 24 else 0.0

        enemy_deltas = []
        try:
            ram = unwrapped.ram
            for i in range(5):
                etype = ram[0x16 + i]
                if etype > 0:
                    ex = ram[0x87 + i]
                    ey = ram[0xCF + i]
                    enemy_deltas.extend([
                        np.clip(ex - (x % 256), -256, 256),
                        np.clip(ey - y, -256, 256),
                    ])
                else:
                    enemy_deltas.extend([256.0, 256.0])
        except Exception:
            enemy_deltas = [256.0] * 10

        vec = np.array([
            x, y, dx, dy,
            info.get("coins", 0), info.get("score", 0), info.get("time", 400),
            info.get("life", 2), info.get("world", 1), info.get("stage", 1),
            1.0 if status == "small" else 0.0,
            1.0 if status == "tall" else 0.0,
            1.0 if status == "fireball" else 0.0,
            1.0 if info.get("flag_get", False) else 0.0,
            near, very_near, dist_next,
            *enemy_deltas,
        ], dtype=np.float32)
        return np.clip(vec / RAM_OBS_MAX, 0.0, 1.0), x, y

    ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+JUMP", "RIGHT+RUN", "R+RUN+JUMP", "JUMP", "LEFT"]
    WINDOW = "Mario AI"
    frame_time = 1.0 / fps if fps > 0 else 0

    total_clears = 0

    for ep in range(episodes):
        # Build raw env — keep unwrapped ref for RAM and rendering
        raw_env = gym_super_mario_bros.make(f"SuperMarioBros-{version}")
        raw_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)
        unwrapped = raw_env.unwrapped  # nes_py env for RAM access
        env = GymV21CompatibilityV0(env=raw_env)

        obs_gym, info = env.reset()
        # Take one NOOP to get real info
        obs_gym, _, _, _, info = env.step(0)

        prev_x, prev_y = info.get("x_pos", 40), info.get("y_pos", 79)
        step = 0
        ep_reward = 0
        done = False

        print(f"Episode {ep+1}/{episodes} | world={info.get('world',1)}-{info.get('stage',1)}")

        while not done:
            t0 = time.time()

            obs, prev_x, prev_y = build_obs(info, prev_x, prev_y, unwrapped)
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            action_int = int(action[0])

            # Debug first 10 steps
            if step < 10:
                print(f"  step={step} x={info.get('x_pos',0)} action={ACTION_NAMES[action_int]} obs[:4]={obs[:4].round(3)}")

            # Frame skip: repeat the predicted action SKIP frames to match training
            # (env_utils_ram.SkipFrame(skip=2)). Model was trained on 2-frame action
            # repeat, so deploy must do the same or dynamics/timing diverge.
            SKIP = 2
            for _ in range(SKIP):
                obs_gym, reward, terminated, truncated, info = env.step(action_int)
                ep_reward += reward
                done = terminated or truncated
                if done:
                    break

            # Render via unwrapped nes_py env directly
            frame = unwrapped.render(mode="rgb_array")
            if frame is not None:
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                # Scale up 3x for visibility
                frame_bgr = cv2.resize(frame_bgr, (768, 720), interpolation=cv2.INTER_NEAREST)
                # Overlay stats
                cv2.putText(frame_bgr, f"x={info.get('x_pos',0)} step={step}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame_bgr, f"action={ACTION_NAMES[action_int]}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                cv2.putText(frame_bgr, f"reward={ep_reward:.0f} clears={total_clears}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                cv2.imshow(WINDOW, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit.")
                    env.close()
                    cv2.destroyAllWindows()
                    return

            if info.get("flag_get", False):
                total_clears += 1
                print(f"  [!] Level cleared! Total: {total_clears}")

            step += 1
            elapsed = time.time() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

        print(f"  Episode done | steps={step} reward={ep_reward:.0f} x={info.get('x_pos',0)}")
        env.close()

    cv2.destroyAllWindows()
    print(f"\nDone. {total_clears} level clears across {episodes} episodes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--version", default="v0", choices=["v0", "v3"])
    parser.add_argument("--fps", type=int, default=30, help="0 = max speed")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        model_path = pick_model()
    model_path = model_path.strip('"').strip("'")

    main(model_path, args.version, args.fps, args.episodes)
