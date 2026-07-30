# Handoff — AI Retro Game Training (Mario)

**Repo:** https://github.com/AutomatronForge/ai_training
**Read first:** README.md, TENSORBOARD_GUIDE.md, this file.

Continuing an AI retro-game training project (PPO + Ollama reward shaping) on an
**EC2 g4dn.2xlarge** (T4 GPU, 8 vCPU / **4 physical cores**, Xeon @ 2.5GHz).

## Current Status (2026-07-30)

- **All training containers currently STOPPED** (idle box). Nothing running.
- Actively training **`ram_v0`** (World 1-1, RAM observations) — reached ~1.6M
  steps with strong results before the latest restart (reward ~3,500, EV ~0.93,
  90+ level clears) on an earlier reward function.
- Latest run added **reward shaping** (coins/score/power-ups/kills/death) — was
  at ~287k steps (1 clear, reached 1-2) when stopped.
- **Claude Code installed** on this box: `~/.local/bin/claude` (v2.1.220).

## Repo Structure

```
games/mario/
├── train/
│   ├── config.py          ← GPU run settings (EDIT to change mode/steps, no rebuild)
│   ├── config_cpu.py      ← CPU run settings (independent of GPU)
│   ├── entrypoint.py      ← reads config.py, dispatches to train script
│   ├── env_utils_ram.py   ← 27-value RAM obs + MarioReward (coins/score/powerups/kills/death)
│   ├── env_utils.py       ← pixel (CNN) env wrappers
│   ├── train_ram.py       ← RAM training + StatsCallback (TensorBoard mario/* metrics)
│   ├── train_v0.py/v3.py  ← pixel training (v0=1-1, v3=all levels)  [NOT yet updated with
│   │                         the save_freq/config/metrics fixes — see "Known gaps"]
│   ├── ollama_coach.py
│   └── viewer.py
├── deploy/
│   ├── showcase.py        ← headless browser showcase (MJPEG on :8081) — WORKING
│   ├── deploy_gym.py      ← runs model in gym emulator (skip=2 fixed)
│   └── deploy_ram.py      ← RetroArch deploy (syntax fixed; still WIP)
├── docker/
│   ├── cuda/  ← GPU (ports 8080 viewer / 6006 TB)   — currently the main run
│   ├── cpu/   ← CPU (ports 8082 viewer / 6008 TB, own config_cpu.py + models_cpu/)
│   └── mac/   ← M3 ARM64
models/          ← GPU-run checkpoints (gitignored; *.zip)
models_cpu/      ← CPU-run checkpoints (isolated from GPU)
infra/
└── cloudserver.yaml  ← CloudFormation template (provisions the EC2 instance;
                         g4dn.* GPU or c7i.* CPU; UserData runs setup_ec2.sh)
setup_ec2.sh     ← EC2 provisioning: Tailscale + Docker + NVIDIA toolkit, then
                    `docker compose up` (GPU compose runs mario-train + a
                    containerized ollama service). No venv/tmux anymore.
```

**Provisioning note:** the `feature/cloudserver-cfn` PR moved EC2 setup to Docker.
`setup_ec2.sh` installs Tailscale (auto-connects as host `cloudserver` via a
Secrets-Manager authkey), Docker + the NVIDIA container toolkit, then brings up the
cuda compose stack. Access is via Tailscale (`http://cloudserver:8080`, `:6006`),
not public ports.

## config.py (GPU run, current)
```python
TRAIN_MODE = "ram_v0"       # ram_v0, ram_v3, pixel_v0, pixel_v3
N_ENVS = 20
TOTAL_TIMESTEPS = 5_000_000
OLLAMA_INTERVAL = 50000     # widened from 5000 so the critic can converge (EV was stalling)
CHECKPOINT_FREQ = 50_000    # timesteps (divided by N_ENVS internally — see fixes)
# LR 3e-4, CLIP 0.2, ENT 0.01, N_STEPS 1024, BATCH 512, N_EPOCHS 8
```

## Fixes applied this session (all on `main`)

1. **Checkpoint save_freq** — SB3 `CheckpointCallback` counts *rollout steps*, not
   timesteps. With N_ENVS=20, `save_freq=50_000` saved at 1M, not 50k. Fixed to
   `max(save_freq // N_ENVS, 1)` in all 4 train scripts. **Checkpoints now save every
   50k timesteps.**
2. **Ollama coach networking** — the coach was failing every query on Linux Docker.
   **Now solved by running Ollama as its own container** in `docker/cuda/docker-compose.yml`
   (service `ollama`, GPU-enabled, reached via `OLLAMA_HOST=ollama` over Docker DNS).
   This came from the `feature/cloudserver-cfn` PR and supersedes the earlier
   `extra_hosts: host.docker.internal` host-gateway approach. (A host-side systemd
   `OLLAMA_HOST=0.0.0.0` drop-in also exists but is no longer needed for this compose.)
3. **config.py wiring** — `train_ram.py` now reads config.py (was hardcoding steps
   at 2M, ignoring the 5M setting) via `_load_config()`.
4. **deploy_gym.py** — added skip=2 to match training `SkipFrame(skip=2)`.
5. **deploy_ram.py** — removed orphaned duplicate lines that broke import.

## New features this session

- **Showcase** (`deploy/showcase.py`): headless MJPEG browser view on :8081, live
  overlay (x, action, reward, **training generation** = checkpoint steps, **REACHED
  FLAG banner**). `--auto-latest` rides the newest checkpoint; `--capture` hunts a
  clean flag-clearing run and loops it. Runs as a **sibling container** off the
  training image (never disturbs training). Uses `deterministic=False` — greedy
  argmax deadlocks (stuck at x=312); the stochastic policy is what clears.
- **Metrics** in `StatsCallback` → TensorBoard under **`mario/*`**: clears_total,
  deaths_total, deaths_per_clear, deaths_per_episode, clear_rate_per_episode,
  max_x_reached, jump_pct, and **per-level clears** (`mario/clears_level/<W-S>`).
  Also stdout `[STATS]` summaries every 50k steps.
- **Reward shaping** (`env_utils_ram.MarioReward`): +2/coin, +0.01/score point,
  +15 power-up gained, −10 lost, +5 enemy kill (inferred from score jump + enemy
  near), +0.5 fireball use, **−25 death**. Fixed weights (not coach-tuned). Changed
  reward scale — judge progress by `mario/*` metrics, not raw ep_rew_mean.
- **CPU variant**: `docker/cpu/` fully isolated (own config_cpu.py, models_cpu/,
  ports 8082/6008, Ollama networking). Image built (`cpu-mario-train-cpu`).

## Key Technical Details

- **No VecNormalize** — obs already normalized 0-1 by RAM_OBS_MAX.
- **RAM obs = 27 values:** x/y/dx/dy, coins/score/time/life/world/stage, status
  flags, pipe proximity (3), enemy deltas (10). **Pipe positions hardcoded for
  World 1-1** — so obs is 1-1-specific; wrong on other levels (matters for v3).
- **Enemy RAM:** x=0x0087+i, y=0x00CF+i, type=0x0016+i (5 slots).
- **`flag_get` is authentic** — library requires flagpole sprite (0x31) + float
  state 3 (sliding). Clearing 1-1 auto-advances to 1-2 (episode continues to death).
- **GPU is idle (~1%) in RAM mode — this is correct.** MlpPolicy is tiny; bottleneck
  is CPU NES emulation. GPU only matters for **pixel** (CnnPolicy) training.

## Docker Commands

```bash
# GPU (main run)
cd games/mario/docker/cuda && sudo docker compose up -d      # ports 8080/6006

# CPU (isolated — safe to run alongside, but shares the same 4 cores)
cd games/mario/docker/cpu  && sudo docker compose up -d      # ports 8082/6008

# Showcase (sibling container, needs a checkpoint in models/)
sudo docker run -d --name mario-showcase -p 8081:8081 \
  -v /home/ubuntu/ai_training/models:/app/models:ro \
  -v /home/ubuntu/ai_training/games/mario/deploy/showcase.py:/app/showcase.py:ro \
  -w /app cuda-mario-train \
  python showcase.py --capture --auto-latest --port 8081 --fps 30
```

SSH port-forward to view from your machine:
```bash
ssh -L 8080:localhost:8080 -L 6006:localhost:6006 \
    -L 8081:localhost:8081 -L 8082:localhost:8082 -L 6008:localhost:6008 ubuntu@<EC2-IP>
```

## Training Speed (measured on this g4dn.2xlarge)

| Config | FPS | Note |
|---|---|---|
| RAM, N_ENVS=20, coach@5k | ~560–650 | 4 cores oversubscribed (load ~14) |
| RAM, N_ENVS=20, coach@50k | ~700 | less coach overhead |

Earlier table's ~5,000 fps figure was aspirational — real RAM throughput here is
~600–700 fps (CPU-bound). Faster CPU RAM training = **more physical cores**, e.g.
`c7i.4xlarge` (8 cores, no GPU, ~$0.71/hr) → set N_ENVS≈16.

## Known gaps / What's Next

1. **Pixel path not updated** — `train_v0.py`/`train_v3.py` + `env_utils.py` still
   have the old checkpoint save_freq bug and lack the config wiring, new metrics,
   and reward shaping. **Audit + port these fixes before running pixel mode.**
2. **GPU→pixel, CPU→RAM split** — to actually use the T4, run pixel (CnnPolicy) on
   GPU and RAM on CPU in parallel. Requires #1 first.
3. **v3 (all 32 levels)** — needs generalizing the hardcoded 1-1 pipe features and
   far more than 5M steps.
4. Phase 4: add Sonic via gym-retro under `games/sonic/`.
5. **CPU compose Ollama style differs** — `docker/cpu/docker-compose.yml` still uses
   the `extra_hosts: host.docker.internal` approach for the coach, while the GPU
   compose now runs a containerized `ollama` service. Align the CPU compose to the
   containerized pattern for consistency when you next touch it.
