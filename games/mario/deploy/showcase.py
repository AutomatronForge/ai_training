"""
showcase.py — Headless browser showcase of a trained Mario RAM model.

Runs the model in the gym-super-mario-bros emulator and streams the gameplay
as MJPEG to a browser on port 8081 (no desktop display needed — works on EC2).
Shows a live overlay: x-position, action, reward, and total level clears.

Designed to run INSIDE the training image (which has torch/SB3/gym/cv2), as a
sibling container so it never disturbs the running training container.

Usage (inside container):
    python showcase.py --model models/mario_ram_v0_ppo_900000_steps.zip
    python showcase.py                # auto-picks the most-trained checkpoint
    python showcase.py --port 8081 --version v0 --fps 30
"""
import argparse
import glob
import os
import re
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ── Frame server (MJPEG over HTTP) ────────────────────────────────────────────
# Self-contained (mirrors train/viewer.py) so no sys.path juggling is needed.
from flask import Flask, Response

app = Flask(__name__)
_frame = None
_lock = threading.Lock()
_stats = {"x": 0, "action": "-", "reward": 0.0, "clears": 0, "episode": 0,
          "step": 0, "generation": 0, "flag": False}


def _set_frame(rgb, stats):
    global _frame
    with _lock:
        _frame = rgb
        _stats.update(stats)


def _mjpeg():
    import cv2
    while True:
        with _lock:
            frame = _frame
        if frame is None:
            disp = np.zeros((480, 512, 3), dtype=np.uint8)
        else:
            disp = cv2.resize(frame, (512, 480), interpolation=cv2.INTER_NEAREST)
        # frame stored RGB → encode as BGR for correct colors in the JPEG
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(1 / 30)


@app.route("/")
def index():
    return """
    <html><head><title>Mario AI — Showcase</title>
    <style>
      *{box-sizing:border-box;margin:0;padding:0}
      body{background:#0a0a0a;color:#eee;font-family:monospace;
           display:flex;flex-direction:column;align-items:center;
           justify-content:center;min-height:100vh;gap:12px}
      h2{font-size:14px;color:#6c7;letter-spacing:3px;text-transform:uppercase}
      #gen{font-size:12px;color:#89a}
      #gen b{color:#fc6}
      .wrap{position:relative}
      img{border:2px solid #1a1a1a;border-radius:6px;image-rendering:pixelated;
          width:640px;height:600px;display:block}
      #flag{position:absolute;top:14px;left:50%;transform:translateX(-50%);
            background:#127a12;color:#fff;font-size:20px;font-weight:bold;
            padding:10px 20px;border-radius:6px;letter-spacing:2px;
            box-shadow:0 0 20px #0f0;display:none}
      #flag.show{display:block}
      #stats{display:flex;gap:22px;font-size:13px;color:#9ab}
      #stats b{color:#fff}
      .clears{color:#6f6}
    </style></head><body>
    <h2>Mario AI &mdash; Trained Model Showcase</h2>
    <div id="gen">training generation: <b id="g">&mdash;</b> steps</div>
    <div class="wrap">
      <img src="/stream">
      <div id="flag">&#128681; REACHED THE FLAG!</div>
    </div>
    <div id="stats">
      <span>x=<b id="x">0</b></span>
      <span>action=<b id="a">-</b></span>
      <span>reward=<b id="r">0</b></span>
      <span class="clears">clears=<b id="c">0</b></span>
      <span>ep=<b id="e">0</b></span>
    </div>
    <script>
      const fmt=n=>n.toLocaleString();
      setInterval(async()=>{const s=await (await fetch('/stats')).json();
        x.textContent=s.x; a.textContent=s.action;
        r.textContent=Math.round(s.reward); c.textContent=s.clears;
        e.textContent=s.episode; g.textContent=fmt(s.generation);
        document.getElementById('flag').className = s.flag ? 'show' : '';
      }, 300);
    </script>
    </body></html>
    """


@app.route("/stream")
def stream():
    return Response(_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats():
    from flask import jsonify
    with _lock:
        return jsonify(_stats)


def start_server(port):
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, threaded=True),
        daemon=True,
    )
    t.start()
    print(f"[showcase] Serving at http://localhost:{port}")


# ── Model / observation (matches training: env_utils_ram.py) ──────────────────
PIPE_X = [224, 400, 616, 790, 1000, 1170, 1364, 1668, 1810, 2060, 2430, 2628]
RAM_OBS_MAX = np.array([
    3000, 255, 10, 10,
    99, 999999, 400, 3, 8, 4,
    1, 1, 1, 1,
    1, 1, 3000,
    256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
], dtype=np.float32)
ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+JUMP", "RIGHT+RUN", "R+RUN+JUMP", "JUMP", "LEFT"]
SKIP = 2  # matches env_utils_ram.SkipFrame(skip=2)


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


def _gen_from_path(path):
    """Extract the training step count ('generation') from a checkpoint name."""
    m = re.search(r"_(\d+)_steps", path or "")
    return int(m.group(1)) if m else 0


def pick_latest_model():
    found = glob.glob("models/*.zip") + glob.glob("../../../models/*.zip")

    def steps(p):
        m = re.search(r"_(\d+)_steps", p)
        return int(m.group(1)) if m else -1

    found = sorted(set(found), key=steps)
    return found[-1] if found else None


def _run_episode(model, make_env, record=False):
    """Play one episode. Returns (cleared, frames) where frames is a list of
    (rgb, stats) tuples if record=True (else empty). Runs at max speed."""
    env, unwrapped = make_env()
    env.reset()
    _, _, _, _, info = env.step(0)
    prev_x, prev_y = info.get("x_pos", 40), info.get("y_pos", 79)
    ep_reward = 0.0
    step = 0
    done = False
    cleared = False
    frames = []

    while not done:
        obs, prev_x, prev_y = build_obs(info, prev_x, prev_y, unwrapped)
        action, _ = model.predict(obs.reshape(1, -1), deterministic=False)
        action_int = int(action[0])
        for _ in range(SKIP):
            _, reward, terminated, truncated, info = env.step(action_int)
            ep_reward += reward
            done = terminated or truncated
            if done:
                break
        if info.get("flag_get", False):
            cleared = True
        if record:
            frame = unwrapped.render(mode="rgb_array")
            if frame is not None:
                frames.append((np.array(frame), {
                    "x": int(info.get("x_pos", 0)),
                    "action": ACTION_NAMES[action_int],
                    "reward": float(ep_reward),
                    "step": step,
                    "flag": cleared,
                }))
        step += 1

    env.close()
    return cleared, frames, int(info.get("x_pos", 0)), step


def capture_and_loop(model_path, version="v0", port=8081, fps=30, auto_latest=False):
    """Hunt (headless, max speed) for one clean flag-clearing episode, record its
    frames, then loop that death-free run on the browser stream forever."""
    from stable_baselines3 import PPO
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

    def make_env():
        raw = gym_super_mario_bros.make(f"SuperMarioBros-{version}")
        raw = JoypadSpace(raw, SIMPLE_MOVEMENT)
        return GymV21CompatibilityV0(env=raw), raw.unwrapped

    model = PPO.load(model_path)
    loaded_path = model_path
    generation = _gen_from_path(loaded_path)
    print(f"[showcase] Capture mode — hunting for a clean clear "
          f"(gen {generation} steps)...")

    start_server(port)

    attempt = 0
    clean_frames = None
    while clean_frames is None:
        attempt += 1
        if auto_latest:  # keep hunting on the newest model
            newest = pick_latest_model()
            if newest and newest != loaded_path and os.path.exists(newest):
                try:
                    model = PPO.load(newest)
                    loaded_path = newest
                    generation = _gen_from_path(loaded_path)
                    print(f"[showcase] Now hunting on gen {generation}")
                except Exception:
                    pass
        cleared, frames, maxx, steps = _run_episode(model, make_env, record=True)
        print(f"[showcase] attempt {attempt}: cleared={cleared} maxx={maxx} steps={steps}")
        # Show a "searching" heartbeat frame so the browser isn't stuck on black
        if frames:
            hb_rgb, _ = frames[-1]
            _set_frame(hb_rgb, {"x": maxx, "action": "SEARCHING", "reward": 0,
                                "clears": 0, "episode": attempt,
                                "generation": generation, "flag": False})
        if cleared:
            clean_frames = frames
            print(f"[showcase] CLEAN CLEAR captured on attempt {attempt} "
                  f"({len(frames)} frames). Looping it on :{port}.")

    # Loop the captured clean run forever.
    frame_time = 1.0 / fps if fps > 0 else 1 / 30
    loop = 0
    while True:
        loop += 1
        for rgb, st in clean_frames:
            _set_frame(rgb, {**st, "clears": 1, "episode": loop,
                             "generation": generation})
            time.sleep(frame_time)
        time.sleep(1.5)  # brief pause on the flagpole before replaying


def main(model_path, version="v0", port=8081, fps=30, auto_latest=False):
    from stable_baselines3 import PPO
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

    print(f"[showcase] Loading model: {model_path}")
    model = PPO.load(model_path)
    loaded_path = model_path
    print(f"[showcase] Loaded. Emulator: SuperMarioBros-{version}")

    start_server(port)

    frame_time = 1.0 / fps if fps > 0 else 0
    total_clears = 0
    episode = 0

    while True:
        episode += 1
        # Ride the improving model: reload the newest checkpoint between episodes
        # so the live stream always plays the best-trained policy available.
        if auto_latest:
            newest = pick_latest_model()
            if newest and newest != loaded_path and os.path.exists(newest):
                try:
                    model = PPO.load(newest)
                    loaded_path = newest
                    print(f"[showcase] Reloaded newest model: {newest}")
                except Exception as e:
                    print(f"[showcase] Reload failed ({e}) — keeping {loaded_path}")
        raw = gym_super_mario_bros.make(f"SuperMarioBros-{version}")
        raw = JoypadSpace(raw, SIMPLE_MOVEMENT)
        unwrapped = raw.unwrapped
        env = GymV21CompatibilityV0(env=raw)

        env.reset()
        _, _, _, _, info = env.step(0)
        prev_x, prev_y = info.get("x_pos", 40), info.get("y_pos", 79)
        ep_reward = 0.0
        step = 0
        done = False
        generation = _gen_from_path(loaded_path)
        reached_flag = False  # latches true for the rest of the episode on a clear

        while not done:
            t0 = time.time()
            obs, prev_x, prev_y = build_obs(info, prev_x, prev_y, unwrapped)
            # deterministic=False: PPO's greedy argmax can deadlock at a single
            # obstacle (e.g. stuck at x=312 vs the flag at ~3160). The policy is
            # stochastic — sampling actions is how it clears the level, matching
            # how it behaved during training.
            action, _ = model.predict(obs.reshape(1, -1), deterministic=False)
            action_int = int(action[0])

            for _ in range(SKIP):  # match training frame-skip
                _, reward, terminated, truncated, info = env.step(action_int)
                ep_reward += reward
                done = terminated or truncated
                if done:
                    break

            frame = unwrapped.render(mode="rgb_array")
            if info.get("flag_get", False) and not reached_flag:
                reached_flag = True  # count each clear once (flag_get stays true after)
                total_clears += 1
                print(f"[showcase] LEVEL CLEARED! ep={episode} total_clears={total_clears}")

            if frame is not None:
                _set_frame(np.array(frame), {
                    "x": int(info.get("x_pos", 0)),
                    "action": ACTION_NAMES[action_int],
                    "reward": float(ep_reward),
                    "clears": total_clears,
                    "episode": episode,
                    "step": step,
                    "generation": generation,
                    "flag": reached_flag,
                })

            step += 1
            sleep = frame_time - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        print(f"[showcase] ep {episode} done | steps={step} "
              f"reward={ep_reward:.0f} x={info.get('x_pos',0)} clears={total_clears}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--version", default="v0", choices=["v0", "v3"])
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--fps", type=int, default=30, help="0 = max speed")
    parser.add_argument("--auto-latest", action="store_true",
                        help="reload the newest checkpoint between episodes")
    parser.add_argument("--capture", action="store_true",
                        help="hunt for one clean flag-clearing run, then loop it")
    args = parser.parse_args()

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        model_path = pick_latest_model()
    if not model_path or not os.path.exists(model_path):
        raise SystemExit("No model found. Pass --model path/to/checkpoint.zip")

    print(f"[showcase] Using model: {model_path}")
    if args.capture:
        capture_and_loop(model_path, args.version, args.port, args.fps, args.auto_latest)
    else:
        main(model_path, args.version, args.port, args.fps, args.auto_latest)
