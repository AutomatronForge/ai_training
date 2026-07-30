# config_cpu.py — settings for the CPU training variant (independent of the GPU run).
# Mounted into the CPU container as /app/config.py, so the same code reads it while
# the GPU container reads the separate games/mario/train/config.py.
#
# Checkpoints go to a separate models_cpu/ dir (see docker/cpu/docker-compose.yml),
# so the two runs never collide.

# "pixel_v0", "pixel_v3", "ram_v0", "ram_v3"
TRAIN_MODE = "ram_v0"

# CPU-only box: 16 physical cores (32 vCPU). N_ENVS=14 measured as the sweet spot
# (~1,980 fps); 24 oversubscribes (load >16) and drops to ~1,660 fps.
N_ENVS = 14

TOTAL_TIMESTEPS = 5_000_000

# Resume the post-fix series from its latest checkpoint (obs format unchanged;
# only the Ollama coach objective + progress_bonus cap changed this restart).
RESUME = True

# PPO hyperparameters
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.01
N_STEPS = 1024
BATCH_SIZE = 512
N_EPOCHS = 8

# Ollama coach interval (steps between LLM queries)
OLLAMA_INTERVAL = 50000

# Checkpoint save frequency (timesteps; divided by N_ENVS internally)
CHECKPOINT_FREQ = 50_000
