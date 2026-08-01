# Reinforcement Learning — Algorithms, Terms & Where They're Used

A plain-language reference, written alongside the Mario project so you can learn the
landscape beyond just our setup. Our Mario agent uses **PPO + a CNN**; this doc puts
that in context with everything else.

---

## Part 1 — Core terms (the vocabulary)

| Term | Plain meaning |
|---|---|
| **Agent** | The thing learning to act (our Mario policy). |
| **Environment (env)** | The world it acts in (the SMB emulator). |
| **Observation / state** | What the agent sees each step (our 84×84 grayscale × 4 frames). |
| **Action** | What it can do (our 7 SIMPLE_MOVEMENT buttons). |
| **Reward** | The scalar feedback signal (our progress + flag-clear + penalties). |
| **Policy (π)** | The agent's strategy: state → action (probabilities). The neural net. |
| **Value function (V)** | Estimate of "how good is this state" (expected future reward). |
| **Episode** | One playthrough start→done (Mario spawn → death or flag). |
| **Rollout** | A batch of steps collected before a learning update. |
| **Return** | Total (discounted) reward over an episode. |
| **Discount (γ)** | How much future reward counts vs now (0.9–0.99 typical). |
| **Exploration vs exploitation** | Trying new things vs using what works. Tuned by **entropy**. |
| **Entropy** | Randomness in the policy. High = explores; →0 = deterministic (can over-narrow/collapse). |
| **On-policy** | Learns ONLY from data the current policy just generated (PPO). Stable, less sample-efficient. |
| **Off-policy** | Can reuse old/replayed data (DQN, SAC). More sample-efficient, trickier to stabilize. |
| **Sample efficiency** | How few environment steps it needs to learn. Off-policy & model-based win here. |
| **Model-free** | Learns to act without predicting the world (PPO, DQN, SAC). |
| **Model-based** | Learns a *model of the world* and plans in it (MuZero, Dreamer). Very sample-efficient. |
| **Actor-critic** | Two-part: "actor" = policy, "critic" = value function. PPO/SAC/A2C are actor-critic. |
| **Replay buffer** | Memory of past transitions to learn from (off-policy only). |
| **CNN feature extractor** | The vision backbone that turns pixels into features (NatureCNN, IMPALA-CNN). NOT an algorithm. |

### Diagnostics you'll see in our logs
- **clip_fraction** (PPO): fraction of updates hitting the clip limit. ~0.5 = updates too big → collapse risk.
- **entropy_loss**: near 0 = policy went deterministic (over-narrowed → our v2/v4 collapses).
- **explained_variance**: how well the critic predicts returns. ~1.0 = models the level well.
- **approx_kl**: how far each update moves the policy. `target_kl` caps it (a safety brake).

---

## Part 2 — The algorithm families

### On-policy (learn from fresh data only)
- **PPO — Proximal Policy Optimization** ⭐ *(what we use)*
  - The workhorse of modern RL. Clips policy updates so they can't move too far → stable & simple.
  - Discrete OR continuous actions. In `stable-baselines3`.
  - Used in: games, robotics, **RLHF for LLMs** (ChatGPT-style alignment historically used PPO).
- **A2C / A3C** — Advantage Actor-Critic (sync / async). PPO's simpler predecessors; mostly superseded.
- **TRPO** — Trust Region Policy Optimization. PPO's stricter, heavier ancestor. Rare now.
- **GRPO** — Group Relative Policy Optimization. A newer PPO variant popular in **LLM reasoning training** (e.g. DeepSeek). Drops the value critic, compares groups of samples.

### Off-policy (reuse replayed data — more sample-efficient)
- **DQN — Deep Q-Network** — the classic that cracked Atari (2015). Value-based, **discrete** actions.
  - **Rainbow** = DQN + 6 improvements bundled; **QR-DQN / C51** = distributional variants; **Ape-X / R2D2** = distributed DQN.
- **DDPG / TD3** — continuous-action, off-policy actor-critic. TD3 fixes DDPG's overestimation.
- **SAC — Soft Actor-Critic** ⭐ for continuous control — adds an entropy bonus for robust exploration.
  - Used in: **robotics, locomotion, drones, autonomous driving research** — anywhere actions are continuous (torques, steering).

### Model-based (learn a world model, plan/imagine in it)
- **MuZero / EfficientZero** — learn a model + plan with Monte-Carlo Tree Search. Beat Atari/Go/Chess/Shogi with one method. Very sample-efficient, complex.
- **Dreamer / DreamerV3** — learn a latent "world model," train the policy inside imagined rollouts. State-of-the-art sample efficiency on pixel tasks; a strong (heavier) alternative to PPO for Mario.

### Distributed / large-scale infrastructure
- **IMPALA (algorithm)** — Importance-Weighted Actor-Learner. Many actors feed one learner; **V-trace** corrects the slightly-stale (off-policy) data. For massive multi-machine training.
  - ⚠️ **Note for our project:** we use the **IMPALA-CNN network**, NOT the IMPALA algorithm. Our algorithm stays PPO. IMPALA-CNN is just a bigger vision backbone we can plug into PPO.
- **SEED RL, Ape-X, R2D2, Gorila** — other distributed-RL frameworks.

---

## Part 3 — Where RL is used (beyond games)

| Domain | Typical algorithms | What RL does |
|---|---|---|
| **Video games / benchmarks** | PPO, DQN/Rainbow, MuZero | Learn to play from pixels/state (our Mario). |
| **Robotics & locomotion** | SAC, PPO, TD3 | Continuous motor control — walking, grasping, balancing. |
| **Autonomous vehicles / drones** | SAC, PPO, model-based | Control & planning under continuous dynamics. |
| **LLM alignment (RLHF)** | PPO, GRPO, DPO* | Fine-tune language models to human preferences. (*DPO is a non-RL alternative.) |
| **Recommendation / ads / ranking** | Contextual bandits, DQN variants | Pick items to maximize long-term engagement. |
| **Operations / control** | PPO, SAC, DQN | Datacenter cooling (DeepMind), power grids, inventory, chip layout, traffic lights. |
| **Finance** | DQN, PPO (research) | Trade execution, portfolio allocation (research-heavy, risky in practice). |
| **Science** | Various | Chemistry synthesis planning, plasma control in fusion reactors (SAC, DeepMind/EPFL). |

---

## Part 4 — Why PPO for Mario (our choices in context)

- **Discrete actions (7 buttons)** → rules out the continuous-only crowd (SAC/TD3/DDPG).
- **Cheap, fast env (emulator)** → sample efficiency matters less, so PPO's on-policy simplicity/stability wins over off-policy complexity.
- **PPO is stable and well-supported** (stable-baselines3) → fewer failure modes to debug.
- **Ceiling reality:** on 1-1 we hit ~97.5% with PPO + small CNN. The remaining deaths are random-enemy timing. Whether a **bigger network** (IMPALA-CNN, our planned v8 test) or a **different algorithm** (Rainbow-DQN, DreamerV3) breaks that ceiling is an open, testable question.

### If we ever wanted to try a different *algorithm* for Mario (future experiments)
- **Rainbow-DQN** — discrete, sample-efficient, strong on Atari; natural PPO alternative here.
- **DreamerV3 / EfficientZero** — model-based; potentially higher ceiling, much bigger setup cost.

---

*Written for the Mario beat-the-game project. Our stack: PPO (stable-baselines3) + NatureCNN/IMPALA-CNN, gym-super-mario-bros, per-level specialists. See the training playbook for the hard-won practical lessons.*
