# Handoff — AI Retro Game Training (Mario)

**Repo:** https://github.com/AutomatronForge/ai_training
**Read first:** README.md, TENSORBOARD_GUIDE.md, this file.

AI retro-game training (PPO + Ollama reward shaping). **Current focus: PIXEL/CNN
training toward all 32 levels — now LIVE on a GPU (T4), after the RAM-observation agent
hit a proven ceiling at World 1-2.**

## Current Status (2026-07-30)

- **On the pixel/CNN path.** The RAM-observation agent (`ram_v0`) **mastered World 1-1**
  but hit a **hard ceiling at 1-2** — established through ~20M steps and 7 falsified
  interventions (pit reward, enemy window+reward, frame-skip, more time, net capacity,
  corrected enemy-obs, long clean training). It reaches ~x850 of 1-2 (flag ~x2500),
  cleared it exactly once (luck). Root cause: the agent can't see full level geometry
  from 29 RAM values. **Conclusion: RAM+MLP topped out; pixel/CNN is the path to 32 levels.**
- **Pixel path ported to full feature-parity** with the RAM framework (config wiring,
  resume, `mario/*` metrics, deep reward shaping, curriculum stage support) and
  **validated on CPU** (correctness only — CnnPolicy on CPU is slow ~765 fps).
- **GPU pixel training is now LIVE.** Running on a **Tesla T4 (16GB) g4dn.2xlarge**
  (8 vCPU): cuda stack up, `pixel_v0` on 1-1, `RESUME=False` (fresh run), `N_ENVS="auto"`
  → **7 envs** (vCPUs-1). Steady **~433 fps**, checkpoints every 50K steps as
  `mario_v0_ppo_*_steps.zip` in `models/`. **ETA ~6.4h for 10M steps** on this box.
- **GPU is expected to sit near-idle (~2% util, 3.3/15GB mem)** — pixel training is
  **CPU-bound on env-stepping**, not GPU-bound. The CNN is tiny; the T4 waits on the 7
  CPU envs. So **N_ENVS scales with vCPU, not GPU.** Adding envs beyond ~vCPUs-1 on a
  given box oversubscribes cores (fps flat/worse, raises SubprocVecEnv-deadlock risk).
  To go faster, get **more vCPUs**, not a bigger GPU: g5.4xlarge (16 vCPU → ~15 envs,
  ~930 fps, ~3h) / g5.8xlarge (32 vCPU → ~31 envs, ~1,920 fps, ~1.4h). (Est. ~62 fps/env,
  linear; real scaling ~10-20% short of that.) N_ENVS change is NOT an obs-shape change —
  can resume the latest checkpoint on a bigger box.
- **NEXT (in-flight):** user is **switching to a larger-vCPU instance** (g5.4xlarge or
  g5.8xlarge) via the console to speed the run up. On relaunch, `N_ENVS="auto"` re-sizes
  to the new box automatically.
- **Early learning signal to watch:** `mario/max_x_reached` on 1-1 plateaued at **~1656**
  (flag ~x3161), 0 clears, deaths climbing — but this was only ~300K/10M steps in (~17 min),
  too early to judge. The pivotal question is whether pixel obs clears 1-1 and breaks the
  RAM 1-2 ceiling; expect an answer well before 10M steps.
- **Hardware note:** the box the framework was *developed* on was CPU-only (16 physical /
  32 vCPU / 61 GB RAM). Current training box is the T4 GPU above (or its g5 successor).

## Repo Structure

```
games/mario/
├── train/
│   ├── config.py          ← GPU run settings (mounted; edit + restart, no rebuild). NOW pixel_v0.
│   ├── config_cpu.py       ← CPU run settings (independent). Currently pixel_v0 prototype.
│   ├── entrypoint.py       ← reads config.py TRAIN_MODE, dispatches to train script
│   ├── env_utils.py        ← PIXEL (CNN) env: GrayScaleResize→VecFrameStack→VecTransposeImage,
│   │                          full MarioReward (coins/score/powerup/kill/death + wall/pit/enemy
│   │                          hazard nudges), tile helpers, `stage` curriculum param
│   ├── env_utils_ram.py     ← 29-value RAM obs + MarioReward + pit detection + `stage` param
│   ├── train_v0.py          ← pixel 1-1 (CnnPolicy) — ported: config/resume/mario metrics/curriculum
│   ├── train_v3.py          ← pixel all-levels (CnnPolicy) — same port
│   ├── train_ram.py         ← RAM training (StatsCallback, mario/* metrics, curriculum)
│   ├── train.py             ← legacy pixel trainer (NOT dispatched; entrypoint ignores it)
│   ├── ollama_coach.py       ← LLM coach; progress_bonus capped ≤0.5, survival-aware prompt
│   └── viewer.py             ← live MJPEG viewer :8080 (pixel path pushes real frames)
├── deploy/  (showcase.py, deploy_gym.py, deploy_ram.py)
├── docker/
│   ├── cuda/   ← GPU stack (ports 8080/6006, mounts config.py, containerized ollama)
│   ├── cpu/    ← CPU stack (ports 8082/6008, mounts config_cpu.py, models_cpu/)
│   └── mac/
models/          ← GPU-run checkpoints (gitignored)
models_cpu/      ← CPU-run checkpoints (+ backup_*/ dirs of superseded runs)
infra/
└── cloudserver.yaml  ← CloudFormation: g4dn.* / g5.* GPU or c7i/c8i CPU; UserData runs setup_ec2.sh
setup_ec2.sh     ← auto-detects GPU vs CPU, installs Tailscale+Docker+NVIDIA toolkit, brings up stack
```

## config.py knobs (both config.py and config_cpu.py support these)

```python
TRAIN_MODE   = "pixel_v0"   # pixel_v0, pixel_v3, ram_v0, ram_v3
N_ENVS       = "auto"       # "auto"/None/0 -> min(32, vCPUs-1); or pin an int
TOTAL_TIMESTEPS = 10_000_000
RESUME       = False        # resume newest models*/mario_<ver>_ppo_*_steps.zip (obs-shape must match)
START_STAGE  = None         # None = 1-1 auto-advance; "1-2" = train that stage directly (curriculum)
NET_ARCH     = None         # MlpPolicy: hidden layers; CnnPolicy: MLP head after conv stack
# LR, CLIP, ENT, N_STEPS, BATCH, N_EPOCHS, OLLAMA_INTERVAL, CHECKPOINT_FREQ
```

- **`N_ENVS="auto"`** self-sizes to the instance (verified in-container: sees host vCPUs
  through Docker). Pixel training is CPU-bound on env-stepping, so ~1 env/vCPU is the ceiling.
- **Obs-shape changes need a fresh run** (`RESUME=False`, back up old checkpoints). Reward/
  coach/stage/hyperparam changes can resume. `NET_ARCH` change = fresh run.

## Key facts / lessons from this session

- **Curriculum works**: `START_STAGE="1-2"` trains directly on `SuperMarioBros-1-2-v0`.
  When a stage is a single-stage env, `life` does NOT decrement on death — StatsCallback
  detects death as *episode-end-without-flag* (dual rule). Success signal per stage =
  `mario/clears_level/<W-S> > 0`, NOT `max_x` (which is level-specific).
- **Ollama coach**: `progress_bonus` capped at 0.5 with a survival-aware prompt (uncapped
  it pinned 1.0 = "rush right" → ran into hazards). Runs as its own container (Docker DNS).
- **SubprocVecEnv can deadlock** (1 core pegged 100%, others Sleeping, logs freeze, no
  traceback) — recover with `docker compose down && up` (NOT `restart`, which serves a
  stale bind-mount inode) resuming the latest checkpoint.
- **Docker file bind-mounts**: editing a mounted `.py` on the host needs `down`/`up` to
  re-resolve (a plain `restart` can keep the old inode).
- **CheckpointCallback save_freq** counts rollout steps → `max(CHECKPOINT_FREQ // N_ENVS, 1)`.

## Launching a GPU box (pixel training)

`infra/cloudserver.yaml` provisions it. GPU options (single-GPU; vCPU is the throughput
knob since env-stepping is CPU-bound — N_ENVS auto-sizes to it):

| Instance | GPU | vCPU | ~$/hr | auto N_ENVS |
|---|---|---|---|---|
| g4dn.xlarge | T4 16GB | 4 | ~0.53 | ~3 |
| g4dn.2xlarge (default) | T4 16GB | 8 | ~0.75 | ~7 |
| g5.xlarge | A10G 24GB | 4 | ~1.01 | ~3 |
| g5.2xlarge | A10G 24GB | 8 | ~1.21 | ~7 |
| g5.4xlarge | A10G 24GB | 16 | ~1.62 | ~15 |
| g5.8xlarge | A10G 24GB | 32 | ~2.45 | ~31 |

(Prices approx us-west-2 on-demand; spot ~30-40%. Verify before launching.)

```bash
# from a machine WITH your AWS creds (this dev box has none):
aws cloudformation create-stack --stack-name mario-gpu --region us-west-2 \
  --template-body file://infra/cloudserver.yaml --capabilities CAPABILITY_NAMED_IAM
# override instance: --parameters ParameterKey=InstanceType,ParameterValue=g5.4xlarge
```
Pre-flight (run with creds): check the **G/VT vCPU quota** (`L-DB2E81BA`, often 0 on new
accounts), and that the template's hardcoded VpcId/SubnetId/SecurityGroupId (`sg-06777192d0a3e6f4f`)/
KeyPair(`ai-test`)/IAM profile(`cloudserver-profile`) still exist. UserData self-provisions
from `main` and starts pixel_v0 on the T4/A10G. Access via Tailscale (`http://cloudserver:8080`
viewer, `:6006` TensorBoard).

## Docker commands

```bash
# GPU (real pixel training)
cd games/mario/docker/cuda && sudo docker compose up -d      # 8080/6006
# CPU (prototype / RAM)
cd games/mario/docker/cpu  && sudo docker compose up -d      # 8082/6008
```

SSH port-forward: `ssh -L 8080:localhost:8080 -L 6006:localhost:6006 -L 8082:localhost:8082 -L 6008:localhost:6008 ubuntu@<IP>`

## What's Next

1. **GPU pixel training is running** (see Current Status). Let `pixel_v0` train on 1-1 to
   competence — the real test of whether pixel obs beats the RAM 1-2 ceiling. Watch
   `mario/max_x_reached` break past ~1656 and `mario/clears_total` go positive.
2. **If pixel clears 1-2** → advance the curriculum (`START_STAGE="1-3"`, then toward v3)
   and re-establish the monitoring loop (health checks + Slack + autonomous decision rules).
3. **v3 (all 32 levels)** with pixel obs + big step budget once the curriculum proves out.
4. Phase 4: add Sonic via gym-retro under `games/sonic/`.

## Known gaps

- Pixel training is **live on GPU (T4)** but **not yet run to competence** — no clears
  of 1-1 yet as of ~300K steps.
- `models/` (GPU) now has `mario_v0_ppo_*_steps.zip` checkpoints from the live run
  (matches the resume glob).
- The autonomous monitoring loop (10-min cron → Slack) is currently **paused/deleted** —
  re-create it now that GPU training is live if you want hands-off monitoring.
- **Watch for SubprocVecEnv deadlock** on the long run (documented failure mode: one core
  pegged, logs freeze) — recover with `docker compose down && up` (NOT `restart`).
