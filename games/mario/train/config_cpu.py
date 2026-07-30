# config_cpu.py — settings for the CPU training variant (independent of the GPU run).
# Mounted into the CPU container as /app/config.py, so the same code reads it while
# the GPU container reads the separate games/mario/train/config.py.
#
# Checkpoints go to a separate models_cpu/ dir (see docker/cpu/docker-compose.yml),
# so the two runs never collide.

# "pixel_v0", "pixel_v3", "ram_v0", "ram_v3"
# PIXEL/CNN prototype on 1-1: validating the ported pixel pipeline on CPU (correctness,
# not competence — CnnPolicy on CPU is slow; real training happens on a GPU box).
TRAIN_MODE = "pixel_v0"

# CPU-only box. CnnPolicy is far heavier per step than the RAM MLP, so keep envs
# modest for the prototype smoke run.
N_ENVS = 8

# Prototype budget — enough to confirm metrics/checkpoints/viewer work, not to train
# to competence. Bump (and move to GPU) for the real 1-1 run.
TOTAL_TIMESTEPS = 5_000_000

# Fresh pixel run (no compatible pixel checkpoint yet). Flip to True to resume the
# newest models_cpu/mario_v0_ppo_*_steps.zip once this series has checkpoints.
RESUME = True

# CnnPolicy: None = SB3 default NatureCNN + default MLP head. (For CnnPolicy, NET_ARCH
# would size only the MLP head after the conv stack.)
NET_ARCH = None

# Curriculum: None = start at 1-1 (auto-advance). Set e.g. "1-2" to train a stage directly.
START_STAGE = None

# PPO hyperparameters (pixel/CNN defaults)
LEARNING_RATE = 2.5e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.05
N_STEPS = 1024
BATCH_SIZE = 1024
N_EPOCHS = 4

# Ollama coach interval (steps between LLM queries)
OLLAMA_INTERVAL = 50000

# Checkpoint save frequency (timesteps; divided by N_ENVS internally)
CHECKPOINT_FREQ = 50_000
