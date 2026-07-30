# config.py — mount this file as a volume so changes take effect on restart
# without rebuilding the Docker image.
#
# docker-compose.yml mounts this as: ./config.py:/app/config.py

# Which training script to run: "pixel_v0", "pixel_v3", "ram_v0", "ram_v3"
TRAIN_MODE = "ram_v0"

# Number of parallel environments
N_ENVS = 20

# Total training steps
TOTAL_TIMESTEPS = 5_000_000

# PPO hyperparameters
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.01
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 8

# Ollama coach interval (steps between LLM queries)
# Widened 5000 -> 50000: at 5k the coach reweighted rewards so often that the
# value function couldn't converge (explained_variance plateaued ~0.3 instead of
# climbing). Fewer, larger-spaced updates give the critic a stable target.
OLLAMA_INTERVAL = 50000

# Checkpoint save frequency
CHECKPOINT_FREQ = 50_000
