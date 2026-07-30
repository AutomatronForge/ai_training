import threading
import time
import numpy as np
from flask import Flask, Response

app = Flask(__name__)

_frame = None
_lock = threading.Lock()
_n_envs = 16


def update_frame(env_idx, frame):
    # Only display env 0 — single large colored view
    if env_idx != 0:
        return
    with _lock:
        global _frame
        _frame = frame


def _generate():
    import cv2
    while True:
        with _lock:
            frame = _frame

        if frame is None:
            blank = np.zeros((84, 84, 3), dtype=np.uint8)
            display = cv2.resize(blank, (504, 504), interpolation=cv2.INTER_NEAREST)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            display = cv2.resize(colored, (504, 504), interpolation=cv2.INTER_NEAREST)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 24)


@app.route("/")
def index():
    return f"""
    <html><head>
    <title>Mario AI Viewer</title>
    <style>
        body {{ background:#0a0a0a; color:#eee; font-family:monospace;
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; min-height:100vh; margin:0; gap:12px }}
        h2 {{ margin:0; font-size:16px; color:#aaa; letter-spacing:2px; text-transform:uppercase }}
        img {{ border:2px solid #222; image-rendering:pixelated; border-radius:4px }}
        p {{ color:#444; font-size:12px; margin:0 }}
    </style>
    </head><body>
    <h2>Mario AI &mdash; {_n_envs} agents training</h2>
    <img src="/stream" width="504" height="504">
    <p>env #0 &bull; viridis colormap &bull; 24fps</p>
    </body></html>
    """


@app.route("/stream")
def stream():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start(n_envs=16):
    global _n_envs
    _n_envs = n_envs
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, debug=False),
        daemon=True,
    )
    t.start()
    print("Viewer running at http://localhost:8080")
