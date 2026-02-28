# SO-ARM101 Reinforcement Learning

Training a 6-DOF robotic arm to reach arbitrary 3D targets using PPO in NVIDIA Isaac Lab.

## Overview

This project trains the [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) robotic arm entirely in simulation using reinforcement learning. The arm learns to move its end effector to random XYZ target positions — no hardcoded trajectories, no inverse kinematics. The policy discovers joint coordination purely from trial and error across 512 parallel environments.

**Stack:** Isaac Lab 2.7.0 · Isaac Sim 5.1.0 · rsl_rl (PPO) · PyTorch · NVIDIA PhysX

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Isaac Lab   │────▶│  PPO Policy  │────▶│  Joint      │
│  Environment │     │  (Actor-     │     │  Position   │
│  512 parallel│◀────│   Critic)    │◀────│  Targets    │
│  arms        │ obs │  [256,128,64]│ act │             │
└─────────────┘     └──────────────┘     └─────────────┘
```

**Observations (16):** joint positions (normalized), joint velocities, end-effector-to-target vector, distance to target

**Actions (6):** joint position targets mapped from [-1, 1] to joint limits

**Reward:** `0.1 × (1 - tanh(5 × distance))` + close bonus (< 1cm) + action smoothness penalty

## Results

| Run | Reward Scale | Entropy | Noise Std | Mean Reward | Notes |
|-----|-------------|---------|-----------|-------------|-------|
| 1   | 1.0         | 0.005   | 1.03      | 105/300     | Critic couldn't learn large values |
| 2   | 0.1         | 0.005   | 0.72      | 28/30       | Best run — reliable reaching |
| 3   | 0.1         | 0.001   | 0.15      | 7/30        | Premature convergence |
| 5   | 0.1         | 0.002   | 0.30      | 5.8/30      | Policy collapse |

Key findings:
- **Reward scaling matters** — the critic learns better predicting 0–30 than 0–300
- **Entropy coefficient is critical** — too high (0.005) keeps noise elevated, too low (0.001) causes premature convergence
- **Observation design** — removed absolute world coordinates in favor of relative features only, reducing obs from 22 to 16

## Files

| File | Description |
|------|-------------|
| `reach_env.py` | Isaac Lab DirectRLEnv — reaching task with randomized targets |
| `train.py` | PPO training script with rsl_rl OnPolicyRunner |
| `so101_robot_cfg.py` | Robot articulation config (joint limits, PD gains) |
| `so101_flat.usd` | Robot USD model converted from URDF |
| `test_mini.py` | Minimal environment test script |

## Training

Requires NVIDIA Isaac Lab 2.7.0+ and Isaac Sim 5.1.0.

```bash
# Train headless (fast)
python train.py

# Train with visualization (slower)
# Set headless=False in train.py
```

Training runs 2000 iterations (~9 minutes on RTX 4060 Ti) across 512 parallel environments.

## What I Learned

1. **PPO entropy tuning** is a balancing act — it controls the exploration/exploitation tradeoff and directly affects whether the policy converges, collapses, or gets stuck
2. **Observation normalization** matters more than hyperparameter tuning — world-frame vs relative coordinates changed results dramatically
3. **Reward shaping** for the critic — scaling rewards to a learnable range (0–30 vs 0–300) had more impact than network architecture changes
4. **Sim-to-real gap** starts in simulation — getting stable training is the first bottleneck before any hardware deployment

## Next Steps

- [ ] Reduced observation space (16 features, relative only) — training in progress
- [ ] Entropy coefficient scheduling (high early → low late)
- [ ] Domain randomization (PD gains, friction, mass)
- [ ] Pick-and-place task built on reaching primitive
- [ ] Sim-to-real transfer to physical SO-ARM101

## License

MIT
