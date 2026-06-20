# amd-rl-lab

This repo is my small RL playground for testing PPO/SAC loops, reward shaping, and training stability on simple environments before trying anything larger.

## Why I built this

RL is hard to debug. A training run might take hours, and you don't know if it worked until the end. I wanted a place to test ideas quickly on simple environments (CartPole, MountainCar) before committing to longer runs.

## What's in here

- PPO implementation (clean, minimal)
- SAC implementation (continuous actions)
- Training logs and reward curves
- Notes on what works and what doesn't

## Current experiments

1. PPO on CartPole-v1 (baseline)
2. PPO on simple trading environment (custom)
3. Reward shaping experiments

## What I'm NOT doing

- Not competing on Atari benchmarks
- Not training for millions of steps
- Not using distributed training

This is for learning and debugging, not for SOTA results.

## Quick start

```bash
pip install -r requirements.txt
python train.py --env CartPole-v1 --algo ppo --steps 50000
```

## Examples

- `examples/cartpole_ppo_run.md` -- a typical CartPole training run
- `examples/simple_trading_env.md` -- custom trading environment
