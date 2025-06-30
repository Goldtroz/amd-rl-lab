# CPU Baseline Benchmarks — AMD RL Lab

All benchmarks run on a single machine without GPU acceleration.

## Environment

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-6000
- **OS:** Ubuntu 22.04
- **Python:** 3.10
- **PyTorch:** 2.3.0 (CPU-only build for baseline)

## PPO — CartPole-v1

| Metric | Value |
|--------|-------|
| Training time (500 episodes) | 42.3 sec |
| Inference time per step | 0.8 ms |
| Peak memory (RSS) | 1.2 GB |
| Final avg reward | 487.2 |
| Steps/sec (rollout) | 3,200 |
| Steps/sec (update) | 12,400 |

## PPO — Breakout (Atari)

| Metric | Value |
|--------|-------|
| Training time (10M steps) | ~18 hours |
| Inference time per step | 2.1 ms |
| Peak memory (RSS) | 3.8 GB |
| Final avg reward | 312.5 |
| Steps/sec (rollout, 8 envs) | 2,800 |
| Steps/sec (update) | 8,600 |

## SAC — HalfCheetah-v4

| Metric | Value |
|--------|-------|
| Training time (1M steps) | ~6.5 hours |
| Inference time per step | 1.4 ms |
| Peak memory (RSS) | 2.1 GB |
| Final avg return | 4,820 |
| Replay buffer sample time | 3.2 ms/batch (256) |

## SAC — Humanoid-v4

| Metric | Value |
|--------|-------|
| Training time (3M steps) | ~22 hours |
| Inference time per step | 1.9 ms |
| Peak memory (RSS) | 4.6 GB |
| Final avg return | 2,150 |
| Replay buffer sample time | 4.8 ms/batch (256) |

---

**AMD GPU benchmark is pending — no access to ROCm hardware yet.**

Expected improvements on RX 7900 XTX:
- Policy/value network updates: 10-20x speedup (matrix ops are GPU-friendly)
- Rollout collection: minimal improvement (env stepping is CPU-bound)
- Mixed precision (fp16): additional 1.5-2x on update phase
- Overall training time reduction: estimated 4-8x for PPO, 3-5x for SAC
