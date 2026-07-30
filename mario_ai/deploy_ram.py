"""
deploy_ram.py — Run trained Mario RAM model against a real emulator.

Requirements:
- RetroArch with Super Mario Bros ROM loaded
- RetroArch network commands enabled:
    Settings > Network > Network Commands = ON (port 55355)
- pip install pyautogui pygetwindow

Usage:
    python deploy_ram.py --model models/mario_ram_v0_final.zip
    python deploy_ram.py --model models/mario_ram_v0_ppo_250000_steps.zip

Controls:
    Q = quit
    P = pause/resume
"""
import argparse
import socket
import struct
import time
import threading
import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

# RetroArch network command port
RETROARCH_HOST = "127.0.0.1"
RETROARCH_PORT = 55355

# NES RAM addresses (same as training)
RAM_MARIO_X       = 0x006D  # x position low byte
RAM_MARIO_X_PAGE  = 0x0086  # x position page (multiply by 256)
RAM_MARIO_Y       = 0x00CE  # y position
RAM_MARIO_STATUS  = 0x0756  # 0=small, 1=tall, 2=fire
RAM_MARIO_LIFE    = 0x075A  # lives remaining
RAM_COINS         = 0x0075  # coin count
RAM_SCORE_1       = 0x07DD  # score (BCD, 6 digits)
RAM_TIME_1        = 0x07F8  # time hundreds
RAM_TIME_2        = 0x07F9  # time tens
RAM_TIME_3        = 0x07FA  # time ones
RAM_WORLD         = 0x075C  # current world (0-indexed)
RAM_STAGE         = 0x075E  # current stage (0-indexed)
RAM_FLAG_GET      = 0x001D  # 1 = flag grabbed
RAM_ENEMY_X       = [0x0087 + i for i in range(5)]  # enemy x positions
RAM_ENEMY_Y       = [0x00CF + i for i in range(5)]  # enemy y positions
RAM_ENEMY_TYPE    = [0x0016 + i for i in range(5)]  # enemy types (0=none)

# SIMPLE_MOVEMENT action → keys mapping
ACTION_KEYS = {
    0: [],                              # NOOP
    1: ["right"],                       # right
    2: ["right", "z"],                  # right + jump (A button = Z in RetroArch default)
    3: ["right", "x"],                  # right + run (B button = X)
    4: ["right", "x", "z"],             # right + run + jump
    5: ["z"],                           # jump in place
    6: ["left"],                        # left
}

# Pipe x positions in World 1-1 (same as training)
PIPE_X = [224, 400, 616, 790, 1000, 1170, 1364, 1668, 1810, 2060, 2430, 2628]

RAM_OBS_MAX = np.array([
    3000, 255, 10, 10,
    99, 999999, 400, 3, 8, 4,
    1, 1, 1, 1,
    1, 1, 3000,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
], dtype=np.float32)


# ── RetroArch RAM reader ─────────────────────────────────────────────────────

class RetroArchRAM:
    """Read NES RAM via RetroArch network commands."""

    def __init__(self, host=RETROARCH_HOST, port=RETROARCH_PORT):
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.1)

    def read_byte(self, address: int) -> int:
        """Read a single byte from NES RAM."""
        cmd = f"READ_CORE_RAM {address:X} 1\n".encode()
        self._sock.sendto(cmd, (self._host, self._port))
        try:
            data, _ = self._sock.recvfrom(64)
            # Response: "READ_CORE_RAM <addr> <hex_byte>"
            parts = data.decode().strip().split()
            return int(parts[2], 16) if len(parts) >= 3 else 0
        except Exception:
            return 0

    def read_bytes(self, address: int, count: int) -> list:
        """Read multiple bytes starting at address."""
        cmd = f"READ_CORE_RAM {address:X} {count}\n".encode()
        self._sock.sendto(cmd, (self._host, self._port))
        try:
            data, _ = self._sock.recvfrom(256)
            parts = data.decode().strip().split()
            return [int(p, 16) for p in parts[2:2+count]]
        except Exception:
            return [0] * count

    def get_game_state(self) -> dict:
        """Read all relevant RAM and return game state dict."""
        x_low  = self.read_byte(RAM_MARIO_X)
        x_page = self.read_byte(RAM_MARIO_X_PAGE)
        x_pos  = x_page * 256 + x_low
        y_pos  = self.read_byte(RAM_MARIO_Y)
        status_val = self.read_byte(RAM_MARIO_STATUS)
        status = ["small", "tall", "fireball"][min(status_val, 2)]
        time_val = (self.read_byte(RAM_TIME_1) * 100 +
                    self.read_byte(RAM_TIME_2) * 10 +
                    self.read_byte(RAM_TIME_3))

        enemy_x = self.read_bytes(RAM_ENEMY_X[0], 5)
        enemy_y = self.read_bytes(RAM_ENEMY_Y[0], 5)
        enemy_t = self.read_bytes(RAM_ENEMY_TYPE[0], 5)

        return {
            "x_pos": x_pos, "y_pos": y_pos,
            "status": status,
            "life": self.read_byte(RAM_MARIO_LIFE),
            "coins": self.read_byte(RAM_COINS),
            "score": 0,
            "time": time_val,
            "world": self.read_byte(RAM_WORLD) + 1,
            "stage": self.read_byte(RAM_STAGE) + 1,
            "flag_get": self.read_byte(RAM_FLAG_GET) == 1,
            "enemy_x": enemy_x,
            "enemy_y": enemy_y,
            "enemy_type": enemy_t,
        }

    def close(self):
        self._sock.close()


# ── Observation builder (identical to training) ───────────────────────────────

class ObsBuilder:
    def __init__(self):
        self._prev_x = 0
        self._prev_y = 0

    def build(self, state: dict) -> np.ndarray:
        x = state["x_pos"]
        y = state["y_pos"]
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._prev_x = x
        self._prev_y = y
        status = state["status"]

        ahead_pipes = [px for px in PIPE_X if px >= x]
        dist_next = (ahead_pipes[0] - x) if ahead_pipes else 3000
        near = 1.0 if dist_next < 48 else 0.0
        very_near = 1.0 if dist_next < 24 else 0.0

        enemy_deltas = []
        for i in range(5):
            etype = state["enemy_type"][i]
            if etype > 0:
                ex = state["enemy_x"][i]
                ey = state["enemy_y"][i]
                enemy_deltas.extend([
                    np.clip(ex - (x % 256), -256, 256),
                    np.clip(ey - y, -256, 256),
                ])
            else:
                enemy_deltas.extend([256.0, 256.0])

        vec = np.array([
            x, y, dx, dy,
            state["coins"], state["score"], state["time"],
            state["life"], state["world"], state["stage"],
            1.0 if status == "small" else 0.0,
            1.0 if status == "tall" else 0.0,
            1.0 if status == "fireball" else 0.0,
            1.0 if state["flag_get"] else 0.0,
            near, very_near, dist_next,
            *enemy_deltas,
        ], dtype=np.float32)
        return np.clip(vec / RAM_OBS_MAX, 0.0, 1.0)


# ── Keyboard controller ───────────────────────────────────────────────────────

class Controller:
    def __init__(self):
        try:
            import pyautogui
            self._pag = pyautogui
            self._available = True
        except ImportError:
            print("[Controller] pyautogui not installed — keypresses disabled")
            self._active_keys = set()
            self._available = False
        self._active_keys = set()

    def press(self, action: int):
        if not self._available:
            return
        new_keys = set(ACTION_KEYS.get(action, []))
        for k in self._active_keys - new_keys:
            self._pag.keyUp(k)
        for k in new_keys - self._active_keys:
            self._pag.keyDown(k)
        self._active_keys = new_keys

    def release_all(self):
        if not self._available:
            return
        for k in self._active_keys:
            self._pag.keyUp(k)
        self._active_keys = set()


# ── Main deploy loop ──────────────────────────────────────────────────────────

def main(model_path: str, fps: int = 30):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import os

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded.")

    ram = RetroArchRAM()
    obs_builder = ObsBuilder()
    controller = Controller()

    paused = False
    running = True

    def keyboard_monitor():
        nonlocal paused, running
        try:
            import keyboard
            keyboard.on_press_key("q", lambda _: setattr(
                threading.current_thread(), "_stop", True))
        except ImportError:
            print("[!] Install 'keyboard' package for Q=quit, P=pause")

    print("\nDeploy started.")
    print("  Make sure RetroArch is open with Mario loaded")
    print("  Settings > Network > Network Commands = ON")
    print("  Press Q to quit\n")

    frame_time = 1.0 / fps
    step = 0
    clears = 0

    try:
        while running:
            t0 = time.time()

            state = ram.get_game_state()
            obs = obs_builder.build(state)
            obs_tensor = obs.reshape(1, -1)

            action, _ = model.predict(obs_tensor, deterministic=True)
            controller.press(int(action[0]))

            if state["flag_get"]:
                clears += 1
                print(f"[!] Level cleared! Total clears: {clears}")

            if step % 60 == 0:
                print(f"Step {step:6d} | x={state['x_pos']:4d} "
                      f"| world={state['world']}-{state['stage']} "
                      f"| status={state['status']} "
                      f"| clears={clears}")

            step += 1
            elapsed = time.time() - t0
            sleep = frame_time - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        controller.release_all()
        ram.close()


if __name__ == "__main__":
    import os
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    model_path = args.model

    # If no model specified or path doesn't exist, show available and prompt
    if not model_path or not os.path.exists(model_path):
        # Search for available models
        search_dirs = ["models", "../models", ".", os.path.dirname(__file__)]
        found = []
        for d in search_dirs:
            found += glob.glob(os.path.join(d, "*.zip"))
        found = sorted(set(found))

        if found:
            print("\nAvailable models:")
            for i, f in enumerate(found):
                print(f"  [{i}] {f}")
            print()
            choice = input("Enter number or full path to model: ").strip()
            if choice.isdigit():
                model_path = found[int(choice)]
            else:
                model_path = choice
        else:
            model_path = input("No models found. Enter full path to model .zip: ").strip()

    # Strip quotes in case user copy-pasted with quotes
    model_path = model_path.strip('"').strip("'")

    if not os.path.exists(model_path):
        print(f"Error: model not found at '{model_path}'")
        exit(1)

    print(f"\nUsing model: {model_path}")
    main(model_path, args.fps)
