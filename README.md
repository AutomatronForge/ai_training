# Mario AI Training — Project Summary

## What We're Building
Training an AI agent to play retro games starting with Super Mario Bros, using PPO (Proximal Policy Optimization) + Ollama LLM reward shaping. End goal: a trained model that can play on real emulators via screen capture.

## Roadmap
- **Phase 1** ✅ Train PPO agent on Mario (running)
- **Phase 2** ✅ Ollama LLM reward shaping (running)
- **Phase 3** 🔲 Deploy to real emulator (screen capture + keyboard injection)
- **Phase 4** 🔲 Multi-game framework (Sonic, Contra, PS1 games)

## Current Status
- Training **SuperMarioBros-v3** (all 32 levels, random each episode) from scratch
- Running on **Windows RTX 3050** via Docker, ~780 fps, ~15 hrs to finish 10M steps
- **Ollama llama3.2:3b** coaching reward weights every 5,000 steps — working correctly
- **9 checkpoints** already saved in `models/`
- Repo: https://github.com/AutomatronForge/ai_training

## Repo Structure
```
AI_TRAINING/
├── mario_ai/          ← Windows CUDA (main)
│   ├── train.py           — current run (v3, 10M steps)
│   ├── train_v0.py        — fine-tune from pretrained 1-1 checkpoint
│   ├── train_v3.py        — full game from scratch
│   ├── env_utils.py       — Mario env wrappers (SkipFrame, MarioReward, etc.)
│   ├── ollama_coach.py    — LLM reward weight adjustment
│   ├── viewer.py          — live MJPEG viewer (Viridis colormap)
│   ├── download_pretrained.py — fetch tsilva HuggingFace checkpoints
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── mario_ai_mac/      ← Mac M3 ARM64 (same code, no CUDA)
├── mario_ai_cpu/      ← CPU only (Steam Deck / no-GPU machines)
├── models/            ← shared checkpoints (gitignored)
└── setup_ec2.sh       ← one-command EC2 setup script
```

## Running Locally

### Windows (Docker + CUDA)
```bash
cd mario_ai
docker compose down && docker compose up -d --build
docker logs mario_ai-mario-train-1 -f
```

### Mac M3 (Docker + MPS)
```bash
cd mario_ai_mac
docker compose up -d --build
```

### View training
- Live viewer: http://localhost:8080
- TensorBoard: http://localhost:6006 (auto-starts with container)

## EC2 Setup (Recommended — 10x faster)
**Recommended instance:** `g4dn.2xlarge` (~$0.75/hr, T4 GPU, 8 vCPUs)
**AMI:** Deep Learning AMI (Ubuntu 22.04) — has CUDA + Python pre-installed

### One-command setup:
```bash
curl -fsSL https://raw.githubusercontent.com/AutomatronForge/ai_training/main/setup_ec2.sh | bash
```

This will:
1. Install Ollama + pull llama3.2:3b
2. Clone the repo
3. Create venv + install all dependencies
4. Launch training in a `tmux` session (persists after SSH disconnect)

### SSH port forwarding (to view from your Mac):
```bash
ssh -L 8080:localhost:8080 -L 6006:localhost:6006 ubuntu@<EC2-IP>
```
Then open http://localhost:8080 and http://localhost:6006 on your Mac.

### To check training on EC2:
```bash
tmux attach -t mario        # watch live output
# Ctrl+B then D to detach without stopping
```

### To run v0 (pretrained fine-tune, faster results):
```bash
cd ~/ai_training/mario_ai
source .venv/bin/activate
python train_v0.py          # downloads tsilva 96%-success checkpoint, fine-tunes
```

### To run v3 (full game, from scratch):
```bash
python train_v3.py
```

## Key Technical Details

### Why Docker on Windows, venv on EC2?
- Windows: Docker handles CUDA driver complexity cleanly
- EC2 Deep Learning AMI: everything pre-installed, venv is simpler + faster (no overhead)

### Ollama Coach
- Runs on host machine (Windows: `host.docker.internal:11434`, EC2: `localhost:11434`)
- Adjusts reward weights every 5,000 steps based on Mario's stats
- Weights clamped to safe ranges to prevent runaway escalation
- Configure host via env var: `OLLAMA_HOST=localhost python train_v3.py`

### Pretrained Checkpoints (tsilva, HuggingFace)
- Trained 50M steps, 96% success rate on World 1-1
- SB3 2.x format, compatible with our setup
- `train_v0.py` downloads and fine-tunes automatically
- Levels available: 1-1, 1-2, 1-4, 2-1, 3-2

### Frame Skip
- `skip=2` (action repeated 2 frames) — finer control for jumping over pipes
- Standard Atari uses skip=4, but Mario's tall pipes need more precision

### Env Versions
- `v0` — always World 1-1 (faster to train, good for testing)
- `v3` — random stage each episode, all 32 levels (current run)

## Next Steps After Training
1. **Check TensorBoard** — look at `ep_rew_mean` curve, should be climbing
2. **Watch for** `[!] Mario cleared a level` in logs
3. **Phase 3** — deploy to RetroArch/FCEUX via screen capture (`mss`) + keyboard injection (`pyautogui`)
4. **Phase 4** — add more games using `gym-retro` (Sonic, Contra, etc.)

## Estimated Training Times
| Hardware | FPS | 10M steps |
|---|---|---|
| RTX 3050 laptop (current) | ~780 | ~15 hrs |
| EC2 g4dn.xlarge (T4) | ~2,000 | ~1.2 hrs |
| EC2 g4dn.2xlarge (T4 + 8 CPU) | ~3,500 | ~45 min |
