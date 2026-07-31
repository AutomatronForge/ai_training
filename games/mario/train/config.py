# config.py — mount this file as a volume so changes take effect on restart
# without rebuilding the Docker image.
#
# docker-compose.yml mounts this as: ./config.py:/app/config.py
# This is the GPU (cuda) run's config. The CPU run uses config_cpu.py.

# Which training script to run: "pixel_v0", "pixel_v3", "ram_v0", "ram_v3"
# PIXEL/CNN on the GPU: this is where the T4 is actually used (CnnPolicy forward
# passes). RAM/MLP is CPU-bound and leaves the GPU idle — don't run it here.
TRAIN_MODE = "pixel_v0"

# Number of parallel environments. "auto" = size to the box (~1 env per vCPU,
# minus 1 for the main proc, capped 32) — so the SAME config works on any g5/g4dn
# size without editing. Set an int to pin it (e.g. 8). NES stepping is CPU-bound,
# so more envs than cores oversubscribes and hurts throughput.
N_ENVS = "auto"

# Total training steps (real 1-1 pixel run; CNN needs more steps than the MLP).
TOTAL_TIMESTEPS = 30_000_000

# Fresh pixel run; flip True to resume the newest models/mario_v0_ppo_*_steps.zip.
# RESTART: was False for the fresh spec-1-1-v2 launch. Flipped True 2026-08-01
# mid-run so any restart (crash recovery, or promotion) RESUMES from the newest
# valid checkpoint + picks up the fixed peak-protector — instead of wiping the
# in-progress climb by starting from scratch.
RESUME = True

# IMPALA-CNN: bigger residual conv features extractor (capacity for many levels;
# the default NatureCNN hit a ceiling and forgot levels). Fresh run when toggled
# (architecture change — old small-net weights don't transfer).
USE_IMPALA = False

# --- SPECIALIST (per-level) training: one model per level, warm-started from the
# previous level's final. Proven approach after generalist attempts all failed. ---
SPECIALIST = True
SPECIALIST_LEVEL = "1-1"      # the level this run trains (drives ckpt tag + final path)
# WARM_START_FROM: load weights from a prior FINAL specialist, then train FRESH on
# SPECIALIST_LEVEL (resets step counter; NOT a resume). "" = scratch (1-1).
# For 1-2: "models/specialists/mario_1-1_final.zip", etc.
WARM_START_FROM = ""

# CnnPolicy: None = SB3 default NatureCNN + MLP head. (NET_ARCH sizes only the
# MLP head after the conv stack for CnnPolicy.)
NET_ARCH = None

# Curriculum: None = start at 1-1 (auto-advance). Set e.g. "1-2" to train a stage directly.
START_STAGE = "1-1"

# All-levels training via CURRICULUM-WEIGHTED mixture: env vector biased toward
# known/easy levels (retain them) with a light tail of harder levels (expand).
# Fixes the catastrophic forgetting seen with a cold uniform all-32 mix.
RANDOM_STAGES = False
CURRICULUM = False

# Metrics run label. "" = auto (persists across RESUME=True, fresh on RESUME=False).
# Set a name (e.g. "pixel-1-1-deathpenalty") to force a new comparable run in Grafana.
# spec-1-1-v2: clean run after the collapse — flag-clear reward + safer hypers.
RUN_NAME = "spec-1-1-v2"

# PPO hyperparameters (pixel/CNN)
# COLLAPSE FIX: the first spec-1-1 hit 71% then cratered to 0% (clip_fraction ~0.5
# = a too-large policy update wrecked the policy at its peak). Lower LR + lower
# entropy make updates gentler and less prone to that blow-up.
LEARNING_RATE = 1.0e-4        # was 2.5e-4 — smaller, safer steps
CLIP_RANGE = 0.2
ENT_COEF = 0.02               # was 0.05 — less random exploration late = fewer collapse-inducing swings
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
