# config.py — mount this file as a volume so changes take effect on restart
# without rebuilding the Docker image.
#
# docker-compose.yml mounts this as: ./config.py:/app/config.py
# This is the GPU (cuda) run's config. The CPU run uses config_cpu.py.

# Which training script to run: "pixel_v0", "pixel_v3", "ram_v0", "ram_v3"
# PIXEL/CNN on the GPU: this is where the T4 is actually used (CnnPolicy forward
# passes). RAM/MLP is CPU-bound and leaves the GPU idle — don't run it here.
TRAIN_MODE = "pixel_v0"

# Number of parallel environments. g4dn.2xlarge = 8 vCPU; NES env-stepping is
# CPU-bound even for pixel, so ~8 envs matches the cores (GPU handles the CNN).
N_ENVS = 8

# Total training steps (real 1-1 pixel run; CNN needs more steps than the MLP).
TOTAL_TIMESTEPS = 10_000_000

# Fresh pixel run; flip True to resume the newest models/mario_v0_ppo_*_steps.zip.
RESUME = False

# CnnPolicy: None = SB3 default NatureCNN + MLP head. (NET_ARCH sizes only the
# MLP head after the conv stack for CnnPolicy.)
NET_ARCH = None

# Curriculum: None = start at 1-1 (auto-advance). Set e.g. "1-2" to train a stage directly.
START_STAGE = None

# PPO hyperparameters (pixel/CNN)
LEARNING_RATE = 2.5e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.05
N_STEPS = 1024
BATCH_SIZE = 1024
N_EPOCHS = 4

# Ollama coach interval (steps between LLM queries)
# Widened 5000 -> 50000: at 5k the coach reweighted rewards so often that the
# value function couldn't converge (explained_variance plateaued ~0.3 instead of
# climbing). Fewer, larger-spaced updates give the critic a stable target.
OLLAMA_INTERVAL = 50000

# Checkpoint save frequency (timesteps; divided by N_ENVS internally)
CHECKPOINT_FREQ = 50_000
