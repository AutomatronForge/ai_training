import json
import threading
import time
import urllib.request
import urllib.error


OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
MODEL = "llama3.2:3b"

DEFAULT_WEIGHTS = {
    "progress_bonus": 0.1,
    "velocity_bonus": 0.05,
    "stuck_penalty": 0.5,
    "jump_bonus": 0.05,
    "stuck_threshold": 90,
}

_weights = dict(DEFAULT_WEIGHTS)
_weights_lock = threading.Lock()
_shared_weights = None  # Set by start() to the Manager dict


def get_weights():
    with _weights_lock:
        return dict(_weights)


def _clamp(new_weights: dict) -> dict:
    float_keys = {"progress_bonus", "velocity_bonus", "stuck_penalty", "jump_bonus"}
    for k in float_keys:
        new_weights[k] = max(0.01, min(1.0, float(new_weights[k])))
    new_weights["stuck_threshold"] = max(30, min(150, int(new_weights["stuck_threshold"])))
    return new_weights


def _ask_ollama(stats: dict) -> dict:
    prompt = f"""You are coaching a Mario AI agent. Analyze these training stats and suggest reward weight adjustments.

Current stats:
- average x_position: {stats['avg_x']:.1f} (max is ~3000 for World 1-1)
- deaths per episode: {stats['deaths_per_ep']:.2f}
- average reward: {stats['avg_reward']:.2f}
- stuck episodes (never moved): {stats['stuck_pct']:.1%}
- jump attempts: {stats['jump_pct']:.1%} of actions

Current reward weights:
{json.dumps(stats['weights'], indent=2)}

Rules:
- If avg_x < 500, increase progress_bonus and jump_bonus to encourage exploration
- If stuck_pct > 0.3, increase stuck_penalty to force movement
- If deaths_per_ep > 5, reduce stuck_threshold so penalty kicks in sooner
- If avg_x > 1500, agent is past the big pipe — reduce jump_bonus, increase progress_bonus
- Keep all float values between 0.01 and 1.0, stuck_threshold between 30 and 150
- Make small adjustments (±0.05 max per step), do not drastically change values

Respond with ONLY valid JSON, no explanation:
{{"progress_bonus": float, "velocity_bonus": float, "stuck_penalty": float, "jump_bonus": float, "stuck_threshold": int}}"""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 100},
    }).encode()

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            text = result.get("response", "").strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                new_weights = json.loads(text[start:end])
                required = {"progress_bonus", "velocity_bonus", "stuck_penalty", "jump_bonus", "stuck_threshold"}
                if required.issubset(new_weights.keys()):
                    return _clamp(new_weights)
    except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        print(f"[Ollama] Error: {e} — keeping current weights")
    return {}


def _coach_loop(stats_fn, interval=5000):
    global _weights
    last_step = 0
    while True:
        time.sleep(10)
        try:
            stats = stats_fn()
            if stats["total_steps"] - last_step < interval:
                continue
            last_step = stats["total_steps"]
            print(f"[Ollama] Querying coach at step {last_step}...")
            new_weights = _ask_ollama(stats)
            if new_weights:
                with _weights_lock:
                    _weights.update(new_weights)
                    snapshot = dict(_weights)
                    # Also update shared Manager dict so subprocesses see new weights
                    if _shared_weights is not None:
                        _shared_weights.update(new_weights)
                print(f"[Ollama] New weights: {snapshot}")
        except Exception as e:
            print(f"[Ollama] Coach loop error: {e}")


def _warmup():
    """Pre-load the model so first real query doesn't time out."""
    print("[Ollama] Warming up model...")
    payload = json.dumps({
        "model": MODEL,
        "prompt": "say ready",
        "stream": False,
        "options": {"num_predict": 3},
    }).encode()
    try:
        req = urllib.request.Request(
            OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        print("[Ollama] Model ready.")
    except Exception as e:
        print(f"[Ollama] Warmup failed: {e} — will retry on first query")


def start(stats_fn, shared_weights=None, interval=5000):
    global _shared_weights
    _shared_weights = shared_weights
    # Warmup in background so training isn't blocked
    warmup_t = threading.Thread(target=_warmup, daemon=True)
    warmup_t.start()
    t = threading.Thread(target=_coach_loop, args=(stats_fn, interval), daemon=True)
    t.start()
    print(f"[Ollama] Coach started — adjusting weights every {interval} steps")
