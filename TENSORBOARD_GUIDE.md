# TensorBoard Charts Guide — Mario AI Training

## How to Use This Guide
- Open http://localhost:6006
- Filter to current run only: type `PPO_19` (or latest) in the "Filter runs" box
- Set Smoothing to 0.6 for cleaner curves
- **Most important charts: `rollout/ep_rew_mean` and `train/explained_variance`**

---

## rollout/ — What Mario is Actually Doing

### `rollout/ep_rew_mean` ⭐ Most Important
**What it measures:** Average total reward per episode (one full life of Mario)  
**What to look for:** Climbing curve — means Mario is getting further and surviving longer  
**Healthy range:** Starts near 0, should climb to 500+ for decent play, 2000+ for level completion  
**Warning signs:** Flat line = stuck, dropping line = collapsing policy  
**RAM vs Pixel:** RAM mode may show higher raw numbers since reward shaping adds bonuses

### `rollout/ep_len_mean`
**What it measures:** Average number of steps per episode before Mario dies or finishes  
**What to look for:** Should increase over time — longer episodes = Mario surviving more  
**Healthy range:** Starts ~100-200 steps, good play = 1000+ steps  
**Why it matters:** Short episodes = Mario dying quickly at the same spot every time

---

## time/ — Training Speed

### `time/fps` ⭐ Watch This
**What it measures:** Environment steps per second across all parallel envs  
**Current values:**
- RAM training (PPO_19): ~1,100 fps
- Pixel training (PPO_4): ~500 fps
- RAM is ~2x faster than pixel

**What to look for:** Should be stable. Drops indicate CPU/GPU bottleneck  
**Our runs comparison:**
| Run | Mode | FPS |
|-----|------|-----|
| PPO_3 | Pixel, 8 envs | ~379 |
| PPO_4 | Pixel, 16 envs | ~525 |
| PPO_19 | RAM, 20 envs | ~1,121 |

### `time/iterations`
**What it measures:** Number of PPO update cycles completed  
**Why it matters:** Each iteration = collected rollout + gradient update. More = more learning

### `time/total_timesteps`
**What it measures:** Total environment steps taken since training started  
**Milestones:**
- 100k — Agent starts learning basic movement
- 500k — Should consistently pass small obstacles
- 2M — RAM mode: should clear World 1-1
- 5M — Pixel mode: should play competently
- 10M — Full run complete

---

## train/ — How Well the Neural Network is Learning

### `train/explained_variance` ⭐ Second Most Important
**What it measures:** How well the value network predicts future rewards (0 = random, 1 = perfect)  
**What to look for:** Should climb from ~0 toward 0.8-0.95  
**Healthy range:** 0.7+ means the network has a good model of the game  
**Warning signs:**
- Near 0 = value network hasn't learned anything yet (normal early on)
- Near 1.0 and staying flat = policy collapsed (happened in our PPO_4 run at ~8M steps)
- Negative = value network is worse than random (very early training)

**Our history:**
- PPO_4 at 8M steps: 0.95 → policy had memorized but stopped improving
- PPO_19 at 184k steps: near 0 → still very early, normal

### `train/entropy_loss`
**What it measures:** How random/exploratory the policy is (more negative = more random)  
**What to look for:** Should stay in range -1.5 to -2.0 for healthy exploration  
**Warning signs:**
- Near 0 = policy collapsed to always doing the same action (our stuck-on-pipe problem)
- Too negative (-2.5+) = too random, not learning patterns
**Why we added `ent_coef=0.05`:** Forces entropy to stay high so Mario keeps exploring

### `train/approx_kl`
**What it measures:** How much the policy changed in the last update (KL divergence)  
**What to look for:** Should stay below 0.05  
**Warning signs:** Above 0.1 = updates too large, policy destabilizing  
**Current:** ~0.01 = very stable, small updates

### `train/clip_fraction`
**What it measures:** % of gradient updates that hit the PPO clip boundary  
**What to look for:** Should stay between 0.05-0.3  
**Warning signs:**
- Near 0 = updates too small, learning slowly
- Above 0.5 = updates too large, consider reducing learning rate
**Current PPO_19:** 0.025 = very low, may want to increase learning rate slightly

### `train/policy_gradient_loss`
**What it measures:** How much the policy is changing (negative = policy improving)  
**What to look for:** Should be small negative number (-0.01 to -0.001)  
**Why it matters:** Large values = big policy swings, instability

### `train/value_loss`
**What it measures:** How wrong the value network's reward predictions are  
**What to look for:** Should decrease over time as value network learns  
**Warning signs:** Increasing value loss = value network getting confused (often from Ollama reward weight changes)

### `train/loss`
**What it measures:** Combined total loss (policy + value + entropy)  
**What to look for:** General downward trend  
**Note:** Can spike when Ollama changes reward weights — this is normal

### `train/learning_rate`
**What it measures:** Current learning rate (constant unless using a scheduler)  
**Our setting:** 3e-4 for RAM, 2.5e-4 for pixel — fixed, no scheduler

### `train/clip_range`
**What it measures:** The PPO clipping threshold (constant)  
**Our setting:** 0.2 — standard PPO default. Lower = more conservative updates

### `train/n_updates`
**What it measures:** Total number of gradient descent steps taken  
**Why it matters:** With `n_epochs=8` (RAM) each iteration = 8 gradient updates

---

## Reading Multiple Runs Together

You have 19 runs. Key ones to compare:

| Run | Description | Steps | Notable |
|-----|-------------|-------|---------|
| PPO_3 | Early pixel, 8 envs | 335k | Entropy collapsed |
| PPO_4 | Pixel, 16 envs, longest | 8.4M | 4.4 hr run, value_loss plateaued |
| PPO_16 | Pixel, 20 envs | 778k | Best pixel run before RAM |
| PPO_19 | **RAM, current** | 184k+ | 2x faster, MlpPolicy |

To compare only PPO_4 (best pixel) vs PPO_19 (RAM):  
Type `PPO_4|PPO_19` in the filter box.

---

## What "Good Training" Looks Like

```
ep_rew_mean:      flat → slow rise → steep rise → plateau
explained_variance: 0 → 0.3 → 0.7 → 0.9 (takes time)
entropy_loss:     -1.9 → stays -1.5 to -2.0 (healthy)
value_loss:       high → drops → stable low
fps:              stable throughout
```

## Red Flags to Watch For

| Signal | Meaning | Fix |
|--------|---------|-----|
| `entropy_loss` near 0 | Policy collapsed | Increase `ent_coef` |
| `explained_variance` near 1.0 + flat reward | Memorized, stopped learning | Reduce learning rate or add noise |
| `value_loss` spiking | Ollama changed weights too aggressively | Tighten Ollama clamp ranges |
| `fps` dropping over time | CPU contention from too many envs | Reduce `N_ENVS` |
| `approx_kl` above 0.1 | Policy updating too fast | Reduce `learning_rate` |
