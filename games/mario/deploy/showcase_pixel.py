"""
showcase_pixel.py — Headless browser showcase of a trained Mario PIXEL/CNN model.

The sibling `showcase.py` is for the OLD RAM model (27-value feature vector).
This one is for the current pixel/CNN specialists: it rebuilds the EXACT training
observation pipeline (grayscale 84x84, 4-frame stack, channel-first) so the CNN
policy sees what it saw in training, while streaming the raw NES COLOR frame to
the browser as MJPEG (port 8081). Live overlay: x, action, reward, clears.

Runs INSIDE the training image (torch/SB3/gym/cv2 present) as a sibling container
so it never disturbs the running trainer.

Usage (inside container):
    python showcase_pixel.py --model models/specialists/mario_1-1_final.zip --stage 1-1
    python showcase_pixel.py --model models/mario_v0_1-1_ppo_26000000_steps.zip --stage 1-1
    python showcase_pixel.py --port 8081 --fps 30 --deterministic
"""
import argparse
import collections
import glob
import os
import re
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from flask import Flask, Response, jsonify

app = Flask(__name__)
_frame = None
_lock = threading.Lock()
_stats = {"x": 0, "action": "-", "reward": 0.0, "clears": 0, "episode": 0,
          "step": 0, "generation": 0, "flag": False, "stage": "-"}

ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+JUMP", "RIGHT+RUN", "R+RUN+JUMP", "JUMP", "LEFT"]
SKIP = 2  # matches env_utils.SkipFrame(skip=2)


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
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(1 / 30)


@app.route("/")
def index():
    return """
    <html><head><title>Mario AI — Specialist Showcase</title>
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
          width:768px;height:720px;display:block}
      #flag{position:absolute;top:14px;left:50%;transform:translateX(-50%);
            background:#127a12;color:#fff;font-size:20px;font-weight:bold;
            padding:10px 20px;border-radius:6px;letter-spacing:2px;
            box-shadow:0 0 20px #0f0;display:none}
      #flag.show{display:block}
      #stats{display:flex;gap:22px;font-size:13px;color:#9ab}
      #stats b{color:#fff}
      .clears{color:#6f6}
    </style></head><body>
    <h2>Mario AI &mdash; Specialist <span id="lvl">&mdash;</span></h2>
    <div id="gen">model: <b id="g">&mdash;</b> steps</div>
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
        lvl.textContent=s.stage;
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
    with _lock:
        return jsonify(_stats)


def start_server(port):
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, debug=False, threaded=True),
        daemon=True,
    )
    t.start()
    print(f"[showcase-pixel] Serving at http://localhost:{port}")


def _gen_from_path(path):
    m = re.search(r"_(\d+)_steps", path or "")
    return int(m.group(1)) if m else 0


def pick_latest_model(stage):
    """Prefer a level-tagged checkpoint, else the final specialist, else any."""
    pats = [
        f"models/mario_v0_{stage}_ppo_*_steps.zip",
        f"../../../models/mario_v0_{stage}_ppo_*_steps.zip",
        f"models/specialists/mario_{stage}_final.zip",
        f"../../../models/specialists/mario_{stage}_final.zip",
    ]
    found = []
    for p in pats:
        found += glob.glob(p)

    def steps(p):
        m = re.search(r"_(\d+)_steps", p)
        return int(m.group(1)) if m else 10**12  # finals sort last (best)
    found = sorted(set(found), key=steps)
    return found[-1] if found else None


class _ObsPipeline:
    """Rebuild the training observation from raw NES frames, per-episode.

    GrayScaleResize(84) -> 4-frame stack (channel-first) -> (1,4,84,84) uint8.
    Matches env_utils: GrayScaleResize + VecFrameStack(4) + VecTransposeImage.
    """
    def __init__(self):
        import cv2
        self._cv2 = cv2
        self.frames = collections.deque(maxlen=4)

    def _gray(self, rgb):
        g = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2GRAY)
        g = self._cv2.resize(g, (84, 84), interpolation=self._cv2.INTER_AREA)
        return g.astype(np.uint8)

    def reset(self, rgb):
        g = self._gray(rgb)
        for _ in range(4):
            self.frames.append(g)
        return self._obs()

    def push(self, rgb):
        self.frames.append(self._gray(rgb))
        return self._obs()

    def _obs(self):
        # stack -> (4,84,84), add batch -> (1,4,84,84)
        return np.stack(self.frames, axis=0)[None, ...]


def main(model_path, stage="1-1", port=8081, fps=30, deterministic=False):
    from stable_baselines3 import PPO
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

    env_id = f"SuperMarioBros-{stage}-v0" if stage else "SuperMarioBros-v0"
    print(f"[showcase-pixel] Loading model: {model_path}")
    model = PPO.load(model_path)
    print(f"[showcase-pixel] Loaded. Emulator: {env_id}  deterministic={deterministic}")

    start_server(port)
    frame_time = 1.0 / fps if fps > 0 else 0
    generation = _gen_from_path(model_path)
    total_clears = 0
    episode = 0

    while True:
        episode += 1
        raw = gym_super_mario_bros.make(env_id)
        raw = JoypadSpace(raw, SIMPLE_MOVEMENT)
        unwrapped = raw.unwrapped
        env = GymV21CompatibilityV0(env=raw)

        env.reset()
        _, _, _, _, info = env.step(0)
        pipe = _ObsPipeline()
        obs = pipe.reset(unwrapped.screen)
        ep_reward = 0.0
        step = 0
        done = False
        reached_flag = False

        while not done:
            t0 = time.time()
            # deterministic=False by default: PPO's stochastic policy is how it
            # clears — greedy argmax can deadlock at a single obstacle.
            action, _ = model.predict(obs, deterministic=deterministic)
            action_int = int(action[0])

            for _ in range(SKIP):
                _, reward, terminated, truncated, info = env.step(action_int)
                ep_reward += reward
                done = terminated or truncated
                if done:
                    break

            obs = pipe.push(unwrapped.screen)
            frame = unwrapped.screen

            if info.get("flag_get", False) and not reached_flag:
                reached_flag = True
                total_clears += 1
                print(f"[showcase-pixel] LEVEL CLEARED! ep={episode} "
                      f"total_clears={total_clears}")

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
                    "stage": stage,
                })

            step += 1
            sleep = frame_time - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

        print(f"[showcase-pixel] ep {episode} done | steps={step} "
              f"reward={ep_reward:.0f} x={info.get('x_pos',0)} clears={total_clears}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--stage", default="1-1", help="level, e.g. 1-1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--fps", type=int, default=30, help="0 = max speed")
    parser.add_argument("--deterministic", action="store_true",
                        help="greedy argmax (default: stochastic, matches training)")
    args = parser.parse_args()

    model_path = args.model
    if not model_path or not os.path.exists(model_path):
        model_path = pick_latest_model(args.stage)
    if not model_path or not os.path.exists(model_path):
        raise SystemExit("No model found. Pass --model path/to/checkpoint.zip")

    print(f"[showcase-pixel] Using model: {model_path}")
    main(model_path, args.stage, args.port, args.fps, args.deterministic)
