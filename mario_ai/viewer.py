import threading
import time
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

_frame = None
_lock = threading.Lock()
_n_envs = 20
_mode = "rgb"  # "rgb" = true color, "viridis" = false color from grayscale


def update_frame(env_idx, frame, mode="rgb"):
    """Accept env 0 frame. mode='rgb' for true color, 'viridis' for grayscale heatmap."""
    if env_idx != 0:
        return
    with _lock:
        global _frame, _mode
        _frame = frame
        _mode = mode


def _generate():
    import cv2
    while True:
        with _lock:
            frame = _frame
            mode = _mode

        if frame is None:
            display = np.zeros((504, 504, 3), dtype=np.uint8)
        else:
            if mode == "viridis":
                # Grayscale input — apply false color
                if frame.ndim == 3 and frame.shape[2] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
                display = cv2.resize(
                    cv2.cvtColor(colored, cv2.COLOR_BGR2RGB),
                    (504, 504), interpolation=cv2.INTER_NEAREST
                )
            else:
                # True RGB — just resize
                rgb = frame if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                display = cv2.resize(rgb, (504, 504), interpolation=cv2.INTER_NEAREST)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 30)


@app.route("/")
def index():
    return f"""
    <html><head>
    <title>Mario AI</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0 }}
        body {{ background: #0a0a0a; color: #eee; font-family: monospace;
                display: flex; flex-direction: column; align-items: center;
                justify-content: center; min-height: 100vh; gap: 10px }}
        h2 {{ font-size: 14px; color: #666; letter-spacing: 3px; text-transform: uppercase }}
        img {{ border: 2px solid #1a1a1a; border-radius: 4px;
               image-rendering: pixelated; width: 512px; height: 512px }}
        p {{ color: #333; font-size: 11px }}
    </style>
    </head><body>
    <h2>Mario AI &mdash; {_n_envs} agents</h2>
    <img src="/stream">
    <p>env #0 &bull; true color &bull; 30fps</p>
    </body></html>
    """


@app.route("/stream")
def stream():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start(n_envs=20):
    global _n_envs
    _n_envs = n_envs
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, debug=False),
        daemon=True,
    )
    t.start()
    print("Viewer running at http://localhost:8080")


def start_ram_render(vec_env, interval=8):
    """For RAM training — render env 0 directly since there's no pixel obs.
    Runs in background thread, calls vec_env.render() every `interval` steps."""
    def _render_loop():
        step = 0
        while True:
            step += 1
            time.sleep(1 / 30)
            if step % interval != 0:
                continue
            try:
                frames = vec_env.env_method("render", indices=[0])
                if frames and frames[0] is not None:
                    import cv2
                    frame = np.array(frames[0])
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        update_frame(0, frame, mode="rgb")
            except Exception:
                pass

    t = threading.Thread(target=_render_loop, daemon=True)
    t.start()
