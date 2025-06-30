# AMD RL Lab

Reinforcement learning experiments running on AMD GPUs via ROCm.

I got tired of NVIDIA being the only viable option for deep RL, so I'm building everything from scratch on my RX 7900 XTX 24GB. It's been... an adventure. ROCm support has come a long way though, and most things actually work now.

## What's in here

- **PPO** — Proximal Policy Optimization, written from scratch. Works on Atari and continuous control tasks.
- **SAC** — Soft Actor-Critic for continuous action spaces. My go-to for robotics-style envs.
- **Custom environments** — Gym/Gymnasium wrappers and a few custom envs I've been messing with.
- **Atari training** — Wrappers for getting Atari games running with frame stacking and all that.
- **RLHF experiments** — Basic reinforcement learning from human feedback. Still very WIP.
- **Multi-agent training** — Self-play and cooperative multi-agent setups.

## Setup

You need ROCm installed and PyTorch built with ROCm support. This is the part that trips people up.

```bash
# Install PyTorch with ROCm (check pytorch.org for the latest)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

pip install -r requirements.txt
```

Verify your GPU is visible:
```python
import torch
print(torch.cuda.is_available())  # yes, it still says "cuda" even on AMD
print(torch.cuda.get_device_name(0))  # should show your AMD GPU
```

## Quick start

```bash
# Train PPO on Breakout
python scripts/train_atari.py --config configs/ppo_atari.yaml

# Train SAC on a continuous control task
python scripts/train_atari.py --config configs/sac_continuous.yaml

# Evaluate a trained agent
python scripts/eval_agent.py --checkpoint checkpoints/ppo_breakout_final.pt
```

## Hardware notes

Running on: **AMD RX 7900 XTX (24GB VRAM)**

ROCm 6.x works pretty well for most RL workloads. The 24GB VRAM is great for larger batch sizes. Some things I've learned:

- `torch.compile()` is hit or miss on ROCm. Sometimes it's faster, sometimes it breaks.
- Mixed precision training works but you need to be careful with certain ops.
- If you get weird segfaults, make sure your ROCm version matches your PyTorch build exactly.
- Memory management is slightly different — watch your VRAM usage in TensorBoard.

## Project structure

```
├── src/
│   ├── ppo_agent.py          # PPO implementation
│   ├── sac_agent.py          # SAC implementation
│   ├── custom_env.py         # Custom Gym environments
│   ├── atari_wrapper.py      # Atari training wrapper
│   ├── rlhf_trainer.py       # RLHF training (WIP)
│   └── multi_agent.py        # Multi-agent training
├── configs/                   # YAML configs for experiments
├── scripts/                   # Training and eval scripts
├── experiments/               # Notes and logs
└── checkpoints/               # Saved models (gitignored)
```

## Why AMD / ROCm

Reinforcement learning is compute-hungry — every training loop burns through millions of environment steps, and the policy/value network updates are pure matrix math that screams on GPUs. The problem? Most RL codebases assume CUDA, and NVIDIA's pricing makes high-VRAM cards inaccessible to independent researchers.

AMD GPUs via ROCm change the equation:

- **Parallel environment simulation:** The RX 7900 XTX's 24GB VRAM lets us run 16+ vectorized environments with large observation spaces, keeping the GPU fed during policy updates while env stepping runs on CPU. This hybrid CPU/GPU pipeline is the sweet spot for RL.
- **Cost-effective training:** A $900 RX 7900 XTX delivers competitive throughput to a $1,600+ RTX 4090 for the matrix operations that dominate RL training.
- **Mixed precision:** fp16/bf16 training on ROCm gives 1.5-2x speedup on policy updates, letting us iterate faster on hyperparameters.
- **Open ecosystem:** ROCm + PyTorch means no vendor lock-in. The same code runs on AMD, and we can contribute fixes upstream.

## AMD GPU Credit Use Plan

1. **Validate on ROCm GPUs** — Run PPO and SAC training end-to-end on AMD hardware, verify convergence matches CPU baseline
2. **Compare CPU vs GPU latency** — Benchmark policy update time, rollout throughput, and end-to-end training time
3. **Test fp16/bf16** — Profile mixed precision stability in PPO (GAE computation) and SAC (Q-value estimation)
4. **Document ROCm issues** — Track `torch.compile` behavior, memory management quirks, and any operator-level failures
5. **Publish benchmarks** — Open results comparing RX 7900 XTX vs CPU baseline, with configs for reproducibility

## License

MIT — do whatever you want with it.
