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

# Bumped to 20M: reverted to skip=2 (skip=1 experiment falsified). Data says 1-2's
# pipe section needs more TRAINING TIME, not more reward/obs hacks (enemy info is
# already in the obs; reach hits ~x1300 halfway and cleared once). Long runway.
TOTAL_TIMESTEPS = 20_000_000

# RESUME=True: the enemy-dy coordinate-frame BUGFIX changes obs VALUES not SHAPE
# (still 29), so the current [256,256] checkpoint resumes fine and now receives a
# CORRECT enemy vertical signal (was garbage: same-level enemies read ~105px below).
RESUME = True

# Bigger policy net: default [64,64] mastered 1-1 but appears capacity-limited on
# 1-2's pipe/enemy navigation. [256,256] is the data-motivated next lever. None = SB3 default.
NET_ARCH = [256, 256]

# CURRICULUM: train DIRECTLY on 1-2 (the fresh bigger net starts here, keeping the
# curriculum). Obs still 29-value. Set None for default 1-1 start.
START_STAGE = "1-2"

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
