# Handoff Prompt for Claude on Mac

Copy and paste the text below into Claude on your Mac:

---

I'm continuing an AI retro game training project. Here's the context:

**Repo:** https://github.com/AutomatronForge/ai_training  
**Read these files first:** README.md, TENSORBOARD_GUIDE.md

## Current Status

- Windows RTX 3050 running `ram_v0` training (5M steps, ~1.5 hrs)
- New clean training run — VecNormalize removed (was causing deploy mismatch)
- Repo restructured into `games/mario/{train,deploy,test,docker}/`

## Repo Structure

```
games/mario/
├── train/          ← all training code
│   ├── config.py   ← EDIT THIS to change mode/steps (no rebuild needed)
│   ├── entrypoint.py
│   ├── env_utils_ram.py   ← 27-value RAM obs, pipe proximity, enemy detection
│   ├── train_ram.py
│   ├── ollama_coach.py
│   └── requirements.txt
├── deploy/
│   ├── deploy_gym.py      ← WORKING — runs model in gym emulator
│   ├── deploy_ram.py      ← WIP — RetroArch deploy
│   └── input_win32.py
├── test/           ← RetroArch diagnostic scripts
└── docker/
    ├── cuda/       ← Windows GPU (run from here)
    ├── mac/        ← M3 ARM64
    └── cpu/        ← no GPU
models/             ← shared checkpoints (all platforms save here)
setup_ec2.sh        ← one-command EC2 setup
```

## config.py (current)
```python
TRAIN_MODE = "ram_v0"       # ram_v0, ram_v3, pixel_v0, pixel_v3
N_ENVS = 20
TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.01
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 8
```

## Key Technical Details

- **No VecNormalize** — removed, obs already normalized 0-1 by RAM_OBS_MAX
- **RAM obs = 27 values:** x/y/dx/dy, coins/score/time/life/world/stage, status flags, pipe proximity (3), enemy positions (10)
- **Pipe positions hardcoded** in env_utils_ram.py for World 1-1
- **Enemy RAM:** x=0x0087+i, y=0x00CF+i, type=0x0016+i (5 enemies)
- **skip=2** frames (not 4) for pipe-clearing jumps
- **Ollama llama3.2:3b** coaches reward weights every 5k steps

## EC2 Setup (one command)

**Recommended:** `g4dn.2xlarge` (~$0.75/hr, T4 GPU, 8 vCPUs)  
**AMI:** Deep Learning AMI Ubuntu 22.04

```bash
curl -fsSL https://raw.githubusercontent.com/AutomatronForge/ai_training/main/setup_ec2.sh | bash
```

After setup, to train full game:
```bash
nano ~/ai_training/games/mario/train/config.py
# Set: TRAIN_MODE = "ram_v3", TOTAL_TIMESTEPS = 10_000_000
tmux attach -t mario  # restart training
```

SSH port forward to watch from Mac:
```bash
ssh -L 8080:localhost:8080 -L 6006:localhost:6006 ubuntu@<EC2-IP>
```

## Docker Commands (new paths)

```bash
# Windows CUDA
cd games/mario/docker/cuda
docker compose up -d --build

# Mac M3
cd games/mario/docker/mac
docker compose up -d --build

# CPU only
cd games/mario/docker/cpu
docker compose up -d --build
```

## Deploy Testing (after training)

```bash
# In test venv
pip install gym-super-mario-bros==7.4.0 nes-py==8.2.1 shimmy[gym-v21]==1.3.0 opencv-python==4.9.0.80 stable-baselines3==2.3.2 "numpy==1.26.4"
python games/mario/deploy/deploy_gym.py
```

## Training Speed

| Hardware | FPS | 5M steps | 10M steps |
|---|---|---|---|
| RTX 3050 laptop | ~1,000 | ~1.4 hrs | ~2.8 hrs |
| EC2 g4dn.xlarge | ~3,000 | ~28 min | ~55 min |
| EC2 g4dn.2xlarge | ~5,000 | ~17 min | ~33 min |

## What's Next

1. Wait for Windows v0 run to finish (~1.5 hrs from restart)
2. Test deploy_gym.py with new checkpoint — should clear 1-1 consistently
3. EC2: train ram_v3 to 10M steps (~33 min on g4dn.2xlarge)
4. Phase 4: add Sonic using gym-retro under games/sonic/
