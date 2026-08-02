import threading
import time
import numpy as np
from flask import Flask, Response, redirect

app = Flask(__name__)

_frame = None
_lock = threading.Lock()
_n_envs = 20
_mode = "rgb"  # "rgb" = true color, "viridis" = false color from grayscale
_shared = None  # Manager dict shared with envs; holds "_color" (0/1) toggle
_coords = {"x": 0, "y": 0, "world": 1, "stage": 1}  # env 0 live position


def update_frame(env_idx, frame, mode="rgb"):
    """Accept env 0 frame. mode='rgb' for true color, 'viridis' for grayscale heatmap."""
    if env_idx != 0:
        return
    with _lock:
        global _frame, _mode
        _frame = frame
        _mode = mode


def update_coords(x, y, world=1, stage=1):
    """Publish env 0's live position (from its info dict) for the overlay."""
    with _lock:
        _coords["x"] = int(x)
        _coords["y"] = int(y)
        _coords["world"] = int(world)
        _coords["stage"] = int(stage)


def _color_on():
    try:
        return bool(_shared is not None and _shared.get("_color", 0))
    except Exception:
        return False


def _generate():
    import cv2
    while True:
        with _lock:
            frame = _frame
            mode = _mode
            cx, cy = _coords["x"], _coords["y"]
            cw, cs = _coords["world"], _coords["stage"]

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

        # Burn env-0 live coords onto the frame (top-left), so the position is
        # visible right on the video for calling out an area.
        label = f"{cw}-{cs}  x={cx}  y={cy}"
        cv2.rectangle(display, (0, 0), (504, 34), (0, 0, 0), -1)
        cv2.putText(display, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )
        time.sleep(1 / 30)


@app.route("/toggle_color", methods=["POST", "GET"])
def toggle_color():
    """Flip full-color mode. Off = fast grayscale (no per-step cost). On = env 0
    streams its raw NES frame in true color (small throughput cost while on)."""
    if _shared is not None:
        try:
            _shared["_color"] = 0 if _shared.get("_color", 0) else 1
        except Exception:
            pass
    return redirect("/")


@app.route("/coords")
def coords():
    from flask import jsonify
    with _lock:
        return jsonify(dict(_coords))


@app.route("/")
def index():
    on = _color_on()
    label = "COLOR: ON (click for fast grayscale)" if on else "COLOR: OFF (click for full color)"
    btn_bg = "#1b6" if on else "#333"
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
        form {{ margin: 0 }}
        button {{ background: {btn_bg}; color: #eee; border: 1px solid #444;
                  font-family: monospace; font-size: 12px; padding: 8px 16px;
                  border-radius: 4px; cursor: pointer; letter-spacing: 1px }}
        button:hover {{ border-color: #888 }}
        #coords {{ font-size: 30px; color: #0f0; letter-spacing: 2px; font-weight: bold }}
    </style>
    </head><body>
    <h2>Mario AI &mdash; {_n_envs} agents &bull; env #0</h2>
    <div id="coords">--</div>
    <img src="/stream">
    <form action="/toggle_color" method="post"><button type="submit">{label}</button></form>
    <p>env #0 &bull; {'full color' if on else 'grayscale (obs)'} &bull; 30fps &bull; live x/y</p>
    <script>
    async function tick() {{
      try {{
        const r = await fetch('/coords'); const s = await r.json();
        document.getElementById('coords').textContent =
          `${{s.world}}-${{s.stage}}   x=${{s.x}}   y=${{s.y}}`;
      }} catch(e) {{}}
    }}
    setInterval(tick, 200); tick();
    </script>
    </body></html>
    """


@app.route("/stream")
def stream():
    return Response(
        _generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def start(n_envs=20, shared_weights=None):
    global _n_envs, _shared
    _n_envs = n_envs
    _shared = shared_weights
    t = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=8080, debug=False),
        daemon=True,
    )
    t.start()
    print("Viewer running at http://localhost:8080")
