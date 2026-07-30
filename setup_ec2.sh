#!/bin/bash
set -e

echo "=== Mario AI EC2 Setup ==="

# 1. Install Ollama
echo "[1/6] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b &
OLLAMA_PID=$!

# 2. Clone repo
echo "[2/6] Cloning repo..."
git clone https://github.com/AutomatronForge/ai_training.git ~/ai_training
cd ~/ai_training/games/mario/train

# 3. Create venv
echo "[3/6] Creating venv..."
python3 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
echo "[4/6] Installing dependencies..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install "numpy<2" --quiet
pip install \
    "gym-super-mario-bros==7.4.0" \
    "nes-py==8.2.1" \
    "stable-baselines3[extra]==2.3.2" \
    "opencv-python-headless==4.9.0.80" \
    "shimmy[gym-v21]==1.3.0" \
    "tensorboard" \
    "flask" --quiet
pip uninstall -y numpy --quiet
pip install --force-reinstall "numpy==1.26.4" --quiet

# 5. Wait for Ollama model pull to finish
echo "[5/6] Waiting for llama3.2:3b pull..."
wait $OLLAMA_PID
echo "Model ready."

# 6. Launch training in tmux
echo "[6/6] Starting training in tmux session 'mario'..."
mkdir -p ~/ai_training/models
mkdir -p tensorboard

export OLLAMA_HOST=localhost

tmux new-session -d -s mario -x 220 -y 50
tmux send-keys -t mario "cd ~/ai_training/games/mario/train && source .venv/bin/activate" Enter
tmux send-keys -t mario "export OLLAMA_HOST=localhost" Enter
tmux send-keys -t mario "tensorboard --logdir tensorboard --host 0.0.0.0 --port 6006 &" Enter
tmux send-keys -t mario "python entrypoint.py" Enter

echo ""
echo "=== Setup complete ==="
echo ""
echo "Training running in tmux. To watch:"
echo "  tmux attach -t mario"
echo ""
echo "To detach from tmux without stopping training:"
echo "  Ctrl+B then D"
echo ""
echo "Ports to forward from your Mac:"
echo "  ssh -L 8080:localhost:8080 -L 6006:localhost:6006 ubuntu@<EC2-IP>"
echo ""
echo "Then open:"
echo "  http://localhost:8080  — live viewer"
echo "  http://localhost:6006  — TensorBoard"
echo ""
echo "To change training mode (edit config.py first):"
echo "  nano ~/ai_training/games/mario/train/config.py"
echo "  # Then restart tmux session"
echo ""
echo "Models save to: ~/ai_training/models/"
