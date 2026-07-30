# Handoff Prompt for Claude on Mac

Copy and paste the text below into Claude on your Mac:

---

I'm continuing an AI retro game training project. Here's the context:

**Repo:** https://github.com/AutomatronForge/ai_training  
**README:** has full project summary  
**TensorBoard guide:** TENSORBOARD_GUIDE.md

## Current Status

Training is running on **Windows RTX 3050** via Docker.  
- Mode: `ram_v0` (World 1-1, RAM observations)
- Steps: 5M target, ~1.5 hrs to complete
- No VecNormalize (removed to fix deploy mismatch)

## What We Built

**Two training approaches:**
1. **Pixel CNN** (`train_v0.py`, `train_v3.py`) — 21MB model, needs screen capture for deploy
2. **RAM MLP** (`train_ram.py`) — 184KB model, reads game memory directly, 10x faster training

**RAM training is the winner:**
- Mario cleared World 1-1 at 233k steps (pixel never cleared in 8M steps)
- 27-value observation: x/y pos, velocity, pipe proximity, 5 enemy positions from RAM
- Ollama llama3.2:3b coaches reward weights every 5k steps
- No VecNormalize — obs already normalized 0-1 by RAM_OBS_MAX

## Key Files

```
mario_ai/
├── config.py          ← CHANGE THIS to switch modes, no rebuild needed
├── env_utils_ram.py   ← RAM obs (27 values), pipe proximity, enemy detection
├── train_ram.py       ← RAM training entry point
├── ollama_coach.py    ← LLM reward shaping
├── deploy_gym.py      ← Deploy against gym emulator (WORKING)
├── deploy_ram.py      ← Deploy against RetroArch (WIP - key injection issues)
└── requirements_deploy.txt
```

## config.py current state
```python
TRAIN_MODE = "ram_v0"      # ram_v0, ram_v3, pixel_v0, pixel_v3
N_ENVS = 20
TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.01
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 8
OLLAMA_INTERVAL = 5000
CHECKPOINT_FREQ = 50_000
```

## RAM Observation Vector (27 values, all 0-1 normalized)
```
[x_pos, y_pos, dx, dy, coins, score, time, life, world, stage,
 is_small, is_tall, is_fireball, flag_get,
 near_pipe, very_near_pipe, dist_to_next_pipe,
 enemy0_dx, enemy0_dy, enemy1_dx, enemy1_dy, enemy2_dx, enemy2_dy,
 enemy3_dx, enemy3_dy, enemy4_dx, enemy4_dy]
```

## Deploy Status

`deploy_gym.py` works — runs model against gym-super-mario-bros emulator directly.  
Mario moves and reaches x≈298 (first pipe) but dies — model needs more training steps.  
Root cause was VecNormalize mismatch (now fixed — removed from training).  

After current 5M v0 run completes, copy checkpoint and test:
```powershell
python deploy_gym.py  # pick mario_ram_v0_ppo_*_steps.zip
```

## What I Need on Mac/EC2

1. **Spin up EC2 `g4dn.2xlarge`** (Deep Learning AMI Ubuntu 22.04, ~$0.75/hr)
2. **Run setup script:**
```bash
curl -fsSL https://raw.githubusercontent.com/AutomatronForge/ai_training/main/setup_ec2.sh | bash
```
3. **Change config for EC2** — edit `mario_ai/config.py` before running:
```python
TRAIN_MODE = "ram_v3"       # full game
TOTAL_TIMESTEPS = 10_000_000
N_ENVS = 20
```
4. **Run training:**
```bash
cd ~/ai_training/mario_ai
source .venv/bin/activate
export OLLAMA_HOST=localhost
python entrypoint.py
```
5. **SSH port forward to watch from Mac:**
```bash
ssh -L 8080:localhost:8080 -L 6006:localhost:6006 ubuntu@<EC2-IP>
```

## Important Notes

- **VecNormalize removed** — was causing deploy mismatch. Don't add it back.
- **Pipe positions hardcoded** in env_utils_ram.py for World 1-1
- **Enemy RAM addresses:** x=0x0087+i, y=0x00CF+i, type=0x0016+i (up to 5 enemies)
- **Ollama** runs on host, container talks to it via `host.docker.internal:11434` (Docker) or `localhost:11434` (EC2 venv)
- **Models shared folder:** `AI_TRAINING/models/` — all checkpoints save here
- **skip=2** (not 4) — needed for pipe-clearing jumps

## Training Speed Estimates

| Hardware | FPS | 10M steps |
|---|---|---|
| RTX 3050 laptop | ~1,000 | ~2.8 hrs |
| EC2 g4dn.xlarge (T4) | ~3,000 | ~55 min |
| EC2 g4dn.2xlarge | ~5,000 | ~33 min |

## Next Steps After EC2 Training

1. Copy `mario_ram_v3_final.zip` to test machine
2. Run `deploy_gym.py` — should clear multiple random levels
3. Fix RetroArch deploy (`deploy_ram.py`) — key injection via win32 SendInput works but x-pos RAM address needs verification with `test_find_x.py`
4. Phase 4: add more games using gym-retro (Sonic, Contra)
