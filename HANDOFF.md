# Handoff Prompt for Claude on Mac

Copy and paste the text below into Claude on your Mac:

---

I'm continuing an AI retro game training project. Here's the context:

**Repo:** https://github.com/AutomatronForge/ai_training  
**README:** has full project summary, read it first.

**What we're doing:**
Training a PPO agent to play Super Mario Bros using Stable Baselines 3 + Ollama LLM reward shaping. Currently training SuperMarioBros-v3 (all 32 levels) on a Windows RTX 3050 laptop via Docker. It's running but slow (~15 hrs for 10M steps).

**What I need help with on Mac:**
1. Spin up an EC2 `g4dn.2xlarge` instance (Deep Learning AMI, Ubuntu 22.04) on AWS
2. SSH into it and run the one-command setup: `curl -fsSL https://raw.githubusercontent.com/AutomatronForge/ai_training/main/setup_ec2.sh | bash`
3. SSH port-forward so I can watch training from my Mac: `ssh -L 8080:localhost:8080 -L 6006:localhost:6006 ubuntu@<EC2-IP>`
4. Verify Ollama is working and training is running in tmux

**Key things to know:**
- Use `train_v3.py` for full game (all 32 levels) or `train_v0.py` to fine-tune from a pretrained 96%-success checkpoint
- Ollama llama3.2:3b is used for reward shaping — install it on EC2 via the setup script
- On EC2 use venv (not Docker) — Deep Learning AMI has everything pre-installed
- Expected speed on g4dn.2xlarge: ~3,500 fps, 10M steps in ~45 minutes vs 15 hours on laptop
- Models are saved to `models/` every 50k steps

**Ports:**
- http://localhost:8080 — live Mario viewer (Viridis colormap)
- http://localhost:6006 — TensorBoard

Please start by helping me launch the EC2 instance from the AWS console.
