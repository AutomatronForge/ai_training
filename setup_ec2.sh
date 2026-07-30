#!/bin/bash
set -e

echo "=== Mario AI EC2 Setup (Docker) ==="

# 0. Base packages
echo "[0/5] Installing base packages (jq, gh)..."
sudo apt-get update -y
sudo apt-get install -y jq

# GitHub CLI (gh) from the official apt repo.
if ! command -v gh >/dev/null 2>&1; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y gh
fi

# Claude Code (native install to ~/.local/bin).
if ! command -v claude >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/claude" ]; then
  curl -fsSL https://claude.ai/install.sh | bash || echo "  (Claude Code install failed — continuing)"
fi

# 1. Tailscale
echo "[1/5] Installing Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh
# Auth key pulled from Secrets Manager — no key in repo. Secret is JSON:
# {"key":"tskey-auth-..."}. Requires the instance role to allow
# secretsmanager:GetSecretValue on this secret.
TS_KEY=$(aws secretsmanager get-secret-value \
  --region us-west-2 \
  --secret-id arn:aws:secretsmanager:us-west-2:502142436846:secret:test/case1-YY7CGT \
  --query SecretString --output text | jq -r '.key')
sudo tailscale up --reset --ssh --hostname cloudserver --authkey "$TS_KEY"
echo "Tailscale up. Reach this box over the tailnet (hostname: cloudserver)."

# 2. Docker + (GPU-only) NVIDIA Container Toolkit
echo "[2/5] Installing Docker..."
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
sudo usermod -aG docker ubuntu || true

# Detect a GPU: nvidia-smi working, or an NVIDIA device on the PCI bus.
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
elif lspci 2>/dev/null | grep -qi nvidia; then
  HAS_GPU=1
fi

if [ "$HAS_GPU" = "1" ]; then
  echo "  GPU detected — installing NVIDIA Container Toolkit + using CUDA stack."
  STACK=cuda
  if ! dpkg -l | grep -q nvidia-container-toolkit; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
  fi
else
  echo "  No GPU detected — using CPU stack (RAM training)."
  STACK=cpu
fi

# 3. Clone repo
echo "[3/5] Cloning repo..."
if [ ! -d ~/ai_training ]; then
  git clone https://github.com/AutomatronForge/ai_training.git ~/ai_training
fi

# 4. Prep shared dirs
echo "[4/5] Preparing directories..."
mkdir -p ~/ai_training/models ~/ai_training/models_cpu

# 5. Launch the selected training stack (training + Ollama, both containerized)
echo "[5/5] Starting Docker stack ($STACK)..."
cd ~/ai_training/games/mario/docker/$STACK
sudo docker compose up -d --build

echo ""
echo "=== Setup complete (stack: $STACK) ==="
echo ""
if [ "$STACK" = "cuda" ]; then
  VIEWER=8080; TB=6006
else
  VIEWER=8082; TB=6008
fi
echo "Stack running in Docker (services: mario training + ollama)."
echo ""
echo "Watch training logs:"
echo "  cd ~/ai_training/games/mario/docker/$STACK && sudo docker compose logs -f"
echo ""
echo "Reach services over Tailscale (hostname: cloudserver):"
echo "  http://cloudserver:$VIEWER  — live viewer"
echo "  http://cloudserver:$TB  — TensorBoard"
echo ""
echo "To change training mode: edit the config, then restart:"
if [ "$STACK" = "cuda" ]; then
  echo "  nano ~/ai_training/games/mario/train/config.py"
else
  echo "  nano ~/ai_training/games/mario/train/config_cpu.py"
fi
echo "  sudo docker compose restart"
echo ""
echo "Models save to: ~/ai_training/models/ (GPU) or models_cpu/ (CPU)"
