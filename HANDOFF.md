# Handoff — AI Retro Game Training (Mario)

**Repo:** https://github.com/AutomatronForge/ai_training
**Read first:** README.md, TENSORBOARD_GUIDE.md, this file.

AI retro-game training (PPO + Ollama reward shaping). **Current focus: PIXEL/CNN
per-level SPECIALISTS toward beating the game — World 1-1 and 1-2 both mastered.**

## Current Status (2026-08-02)

**Approach = one small-net (NatureCNN) PPO specialist per level**, cold-started per level
(warm-start from a *converged* prior level FREEZES on a structurally different layout — see
lessons). Proving out World 1 (1-1→1-4) as PoC before all 32. Deaths allowed (beat-the-game).
Hardware now: **A10G (23GB), 32 vCPU, 124GB RAM**; `N_ENVS="auto"` → 31 envs, ~1,090–1,750 fps.

**Deliverables (all in `models/specialists/`, backed up to gdrive:mario_ai_backups):**
- **1-1: `mario_1-1_final.zip` @ 97.5%** — mastered. Small-net ceiling ~97.5% (remaining
  ~2.5% = scattered random-enemy-timing deaths). **DEPLOY STOCHASTIC** (argmax deadlocks: 0/10 vs 10/10).
- **1-2: `mario_1-2_final.zip` @ 89% peak (recent ~82-87%)** — MASTERED at the user's 80%×3 bar.
  This was the known-hard bottleneck; solved this session (see below). Small-net 1-2 ceiling ~89%.

**Stopped here (2026-08-02, user call).** Training container is idle/stoppable.

**1-3 IN PROGRESS (`spec-1-3-cold`, COLD start).** Cold-started (warm-start from a converged
prior level FREEZES — proven on 1-2). Config left ready to RESUME 1-3: `SPECIALIST_LEVEL/
START_STAGE="1-3"`, `WARM_START_FROM=""`, `RESUME=False`, `RUN_NAME="spec-1-3-cold"`, small net,
lr 1e-4/ent 0.03/target_kl 0.02. Reached ~7M steps: **0 clears yet**, avg_max_x ~467, best_x ~1524
(~60% of the level; flag x=2560), depth climbing monotonically (not stalled). **1-3 death histogram:
dominant EARLY wall at x~300-400 (~30k deaths), secondary ~600-700** (the first gap/platform jump).
No 1-3 model banked yet (0 clears). To continue: `docker compose up -d` resumes the fresh cold run
(RESUME=False + no checkpoint = starts over — set `RESUME=True` to continue existing 1-3 checkpoints
if any exist, else it restarts cold). **If it stalls at x~300-400**, seed `CHECKPOINTS_BY_LEVEL["1-3"]
= [300,400,600,700,...]` from the histogram and/or add a `NOJUMP_BANDS_BY_LEVEL["1-3"]` if the
`:8080` coord overlay shows a jump-into-chamber trap (the 1-2 playbook).

### How 1-2 was solved this session (the hard level)
1. **Cold start + small net** (`USE_IMPALA=False`). An IMPALA-CNN A/B on 1-1 (v8) LOST —
   bigger net was less sample-efficient, no ceiling break. Ceiling is the reactive-CNN
   *method*, not capacity. Keep `USE_IMPALA=False`.
2. **Checkpoint-crossing bonus** (`CHECKPOINTS_BY_LEVEL`, +8 one-time per measured death-wall)
   — seeded 1-2 from its death histogram: 450/600/750/900/1050/1200/1500/1950/2308 (flag x2560).
3. **Stuck-escape retreat window** (`STUCK_ESCAPE_WINDOW=60`) — once wedged, briefly waive the
   stuck penalty + tolerate a backward step so Mario can back up to run-jump a pipe. Farm-proof.
4. **No-jump band** (`NOJUMP_BANDS_BY_LEVEL={"1-2":[(930,1040)]}`, `GROUND_Y`, `RUN_LOW_BONUS`)
   — the KEY fix. At x~978 there's a CHAMBER Mario kept JUMPING UP INTO and bouncing on the
   ceiling; the generic wall/pit jump-nudge was luring him in. In the band we suppress ALL jump
   rewards and reward forward progress while grounded (run under, don't jump in). This broke the
   x~978 logjam (0% → clears), moving the wall to an enemy at x~1180, which it then learned.
5. **Collapse + recovery**: at ~14.95M steps 1-2 over-narrowed (entropy→0, frozen at x198,
   recent 63%→0%). Recovered by warm-starting from the peak-protector's banked 62.8% best with
   anti-collapse guardrails (lr 3e-5, ent 0.05, target_kl 0.02) → climbed back to 89%.

- **Live coordinate overlay** added to `viewer.py` (`:8080`): big x/y readout + `1-2` label
  burned on the frame, plus `/coords` JSON. Used to diagnose the chamber vs pit visually.
- **Peak-protector** (`BestCheckpointCallback` in `train_v0.py`) saves best-by-recent-clear%
  to `models/best/` + `models/specialists/*_best_fallback.zip`, atomic, prune-immune, seeds
  high-water from on-disk `.pct` so a restart can't clobber a better model. This is what made
  the collapse recoverable.
- **Off-box backups**: `backup_models.sh` (repo root) → rclone COPY (append-only, never deletes
  remote) of models/best + specialists + archive to `gdrive:mario_ai_backups`. Run every ~20 min.

## Repo Structure

```
games/mario/
├── train/
│   ├── config.py          ← GPU run settings (mounted; edit + restart, no rebuild). NOW pixel_v0.
│   ├── config_cpu.py       ← CPU run settings (independent). Currently pixel_v0 prototype.
│   ├── entrypoint.py       ← reads config.py TRAIN_MODE, dispatches to train script
│   ├── env_utils.py        ← PIXEL (CNN) env: GrayScaleResize→VecFrameStack→VecTransposeImage,
│   │                          full MarioReward (coins/score/powerup/kill/death + wall/pit/enemy
│   │                          hazard nudges), tile helpers, `stage` curriculum param
│   ├── env_utils_ram.py     ← 29-value RAM obs + MarioReward + pit detection + `stage` param
│   ├── train_v0.py          ← pixel 1-1 (CnnPolicy) — ported: config/resume/mario metrics/curriculum
│   ├── train_v3.py          ← pixel all-levels (CnnPolicy) — same port
│   ├── train_ram.py         ← RAM training (StatsCallback, mario/* metrics, curriculum)
│   ├── train.py             ← legacy pixel trainer (NOT dispatched; entrypoint ignores it)
│   ├── ollama_coach.py       ← LLM coach; progress_bonus capped ≤0.5, survival-aware prompt
│   └── viewer.py             ← live MJPEG viewer :8080 (pixel path pushes real frames)
├── deploy/  (showcase.py, deploy_gym.py, deploy_ram.py)
├── docker/
│   ├── cuda/   ← GPU stack (ports 8080/6006, mounts config.py, containerized ollama)
│   ├── cpu/    ← CPU stack (ports 8082/6008, mounts config_cpu.py, models_cpu/)
│   └── mac/
models/          ← GPU-run checkpoints (gitignored)
models_cpu/      ← CPU-run checkpoints (+ backup_*/ dirs of superseded runs)
infra/
└── cloudserver.yaml  ← CloudFormation: g4dn.* / g5.* GPU or c7i/c8i CPU; UserData runs setup_ec2.sh
setup_ec2.sh     ← auto-detects GPU vs CPU, installs Tailscale+Docker+NVIDIA toolkit, brings up stack
```

## config.py knobs (both config.py and config_cpu.py support these)

```python
TRAIN_MODE   = "pixel_v0"   # pixel_v0, pixel_v3, ram_v0, ram_v3
N_ENVS       = "auto"       # "auto"/None/0 -> min(32, vCPUs-1); or pin an int
TOTAL_TIMESTEPS = 10_000_000
RESUME       = False        # resume newest models*/mario_<ver>_ppo_*_steps.zip (obs-shape must match)
START_STAGE  = None         # None = 1-1 auto-advance; "1-2" = train that stage directly (curriculum)
NET_ARCH     = None         # MlpPolicy: hidden layers; CnnPolicy: MLP head after conv stack
# LR, CLIP, ENT, N_STEPS, BATCH, N_EPOCHS, OLLAMA_INTERVAL, CHECKPOINT_FREQ
```

- **`N_ENVS="auto"`** self-sizes to the instance (verified in-container: sees host vCPUs
  through Docker). Pixel training is CPU-bound on env-stepping, so ~1 env/vCPU is the ceiling.
- **Obs-shape changes need a fresh run** (`RESUME=False`, back up old checkpoints). Reward/
  coach/stage/hyperparam changes can resume. `NET_ARCH` change = fresh run.

## Key facts / lessons from this session

- **Curriculum works**: `START_STAGE="1-2"` trains directly on `SuperMarioBros-1-2-v0`.
  When a stage is a single-stage env, `life` does NOT decrement on death — StatsCallback
  detects death as *episode-end-without-flag* (dual rule). Success signal per stage =
  `mario/clears_level/<W-S> > 0`, NOT `max_x` (which is level-specific).
- **Ollama coach**: `progress_bonus` capped at 0.5 with a survival-aware prompt (uncapped
  it pinned 1.0 = "rush right" → ran into hazards). Runs as its own container (Docker DNS).
- **SubprocVecEnv can deadlock** (1 core pegged 100%, others Sleeping, logs freeze, no
  traceback) — recover with `docker compose down && up` (NOT `restart`, which serves a
  stale bind-mount inode) resuming the latest checkpoint.
- **Docker file bind-mounts**: editing a mounted `.py` on the host needs `down`/`up` to
  re-resolve (a plain `restart` can keep the old inode).
- **CheckpointCallback save_freq** counts rollout steps → `max(CHECKPOINT_FREQ // N_ENVS, 1)`.

## Launching a GPU box (pixel training)

`infra/cloudserver.yaml` provisions it. GPU options (single-GPU; vCPU is the throughput
knob since env-stepping is CPU-bound — N_ENVS auto-sizes to it):

| Instance | GPU | vCPU | ~$/hr | auto N_ENVS |
|---|---|---|---|---|
| g4dn.xlarge | T4 16GB | 4 | ~0.53 | ~3 |
| g4dn.2xlarge (default) | T4 16GB | 8 | ~0.75 | ~7 |
| g5.xlarge | A10G 24GB | 4 | ~1.01 | ~3 |
| g5.2xlarge | A10G 24GB | 8 | ~1.21 | ~7 |
| g5.4xlarge | A10G 24GB | 16 | ~1.62 | ~15 |
| g5.8xlarge | A10G 24GB | 32 | ~2.45 | ~31 |

(Prices approx us-west-2 on-demand; spot ~30-40%. Verify before launching.)

```bash
# from a machine WITH your AWS creds (this dev box has none):
aws cloudformation create-stack --stack-name mario-gpu --region us-west-2 \
  --template-body file://infra/cloudserver.yaml --capabilities CAPABILITY_NAMED_IAM
# override instance: --parameters ParameterKey=InstanceType,ParameterValue=g5.4xlarge
```
Pre-flight (run with creds): check the **G/VT vCPU quota** (`L-DB2E81BA`, often 0 on new
accounts), and that the template's hardcoded VpcId/SubnetId/SecurityGroupId (`sg-06777192d0a3e6f4f`)/
KeyPair(`ai-test`)/IAM profile(`cloudserver-profile`) still exist. UserData self-provisions
from `main` and starts pixel_v0 on the T4/A10G. Access via Tailscale (`http://cloudserver:8080`
viewer, `:6006` TensorBoard).

## Docker commands

```bash
# GPU (real pixel training)
cd games/mario/docker/cuda && sudo docker compose up -d      # 8080/6006
# CPU (prototype / RAM)
cd games/mario/docker/cpu  && sudo docker compose up -d      # 8082/6008
```

SSH port-forward: `ssh -L 8080:localhost:8080 -L 6006:localhost:6006 -L 8082:localhost:8082 -L 6008:localhost:6008 ubuntu@<IP>`

## What's Next

1. **Continue 1-3** (`spec-1-3-cold`, cold, in progress ~7M steps, 0 clears yet). `docker compose
   up -d` to resume. Watch avg_max_x/clear% break past the x~300-400 early wall. If it stalls,
   seed `CHECKPOINTS_BY_LEVEL["1-3"]` from the histogram (300/400/600/700) + a no-jump band if the
   `:8080` overlay shows a jump-trap. Promote at 80%×3 (awaiting user go), save `mario_1-3_final.zip`.
2. **1-4, then all 32** with the same per-level recipe (cold + checkpoint-bonus + per-level
   no-jump bands where a chamber/alcove traps the jump-nudge). Then a router (32 models → 1 agent).
3. Phase 4: add Sonic via gym-retro under `games/sonic/`.

## Reward-shaping mechanisms (per-level, in `env_utils.py` MarioReward)

- `FLAG_CLEAR_BONUS=300` (dominant terminal reward), `DEATH_PENALTY=40`+progress-scale,
  `POWERDOWN_PEN=20`, `COIN/SCORE/POWERUP/KILL` bonuses, hazard-aware wall/pit/enemy jump nudges.
- `CHECKPOINTS_BY_LEVEL` — one-time +8 per measured death-wall x (from the death histogram). Per-level.
- `STUCK_ESCAPE_WINDOW=60` — retreat-to-jump escape for pipe wedges (waive stuck penalty + tolerate
  one backward step; forward progress closes it; backtracking never rewarded).
- `NOJUMP_BANDS_BY_LEVEL` + `GROUND_Y` + `RUN_LOW_BONUS` — x-bands where jumping traps the agent in a
  chamber; suppress jump rewards there and reward running forward while grounded. **The 1-2 unlock.**
- To find a level's walls: query `metrics/mario.db` `episodes` (death_x / max_x / status) for the run.

## Anti-collapse + recovery playbook

- Collapse signatures: `clip_fraction→~0.5` = updates too big (lower lr); `clip→~0.07`+`entropy→0`
  = policy over-narrowed/frozen (raise entropy / target_kl). Both cliff recent-clear% to 0%.
- Recover: warm-start from the peak-protector's banked best (`models/best/mario_v0_<lvl>_best.zip`,
  its `.pct` shows the banked clear%) with `RESUME=False`, guardrails **lr 3e-5, ent 0.05,
  target_kl 0.02**, a fresh `RUN_NAME`. Stage the best to a stable path first (a live prune could
  touch `models/best/`). This recovered 1-2 from a full freeze back to 89%.

## Known gaps / cautions

- **`config.py` is left on the 1-3 COLD run**: `SPECIALIST_LEVEL/START_STAGE="1-3"`,
  `WARM_START_FROM=""`, `RESUME=False`, `RUN_NAME="spec-1-3-cold"`, lr 1e-4/ent 0.03. `up -d`
  resumes 1-3 (fresh cold — no 1-3 checkpoints banked yet). For a DIFFERENT level, repoint
  SPECIALIST_LEVEL/START_STAGE/RUN_NAME (+ WARM_START_FROM="" for cold).
- Promotion bar per user = **80%×3** (the cron text still says 70%; user raised it).
- `models/` is gitignored — all model deliverables live only on-box + `gdrive:mario_ai_backups`.
  Recovery source `mario_1-2_recover_src.zip` is a copy of the banked 62.8% best (kept for provenance).
- SubprocVecEnv deadlock recovery = `docker compose down && up` (NOT `restart`).
