# Handoff — AI Retro Game Training (Mario)

**Repo:** https://github.com/AutomatronForge/ai_training
**Read first:** README.md, TENSORBOARD_GUIDE.md, this file.

AI retro-game training (PPO + Ollama reward shaping). **Current focus: PIXEL/CNN
training toward all 32 levels, after the RAM-observation agent hit a proven ceiling
at World 1-2.**

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
- **Ready to launch a GPU box** for real pixel training. GPU `config.py` is set to
  `pixel_v0`, `N_ENVS="auto"`.
- **Hardware note:** the box this was developed on is **CPU-only** (16 physical cores /
  32 vCPU / 61 GB RAM) — NOT the g4dn the older handoffs assumed. Real pixel training
  needs a GPU instance (see Launching a GPU box).

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

1. **Launch a GPU box** (g5.2xlarge or g5.4xlarge recommended) and train pixel `pixel_v0`
   on 1-1 to competence — the real test of whether pixel obs beats the RAM 1-2 ceiling.
2. **If pixel clears 1-2** → advance the curriculum (`START_STAGE="1-3"`, then toward v3)
   and re-establish the monitoring loop (health checks + Slack + autonomous decision rules).
3. **v3 (all 32 levels)** with pixel obs + big step budget once the curriculum proves out.
4. Phase 4: add Sonic via gym-retro under `games/sonic/`.

## Known gaps

- Pixel training only validated for correctness on CPU; **not yet run on GPU / to competence.**
- `models/` (GPU) will be empty until a GPU box runs; pixel checkpoints save as
  `mario_v0_ppo_*_steps.zip` (matches the resume glob).
- The autonomous monitoring loop (10-min cron → Slack) is currently **paused/deleted** —
  re-create it once GPU training is live if you want hands-off monitoring.
