# Experiment Notes

## 2026-05-28 — First PPO run

Got PPO running on Atari Breakout. Initial results are... not great but it's learning.

```
Episode 100 | Reward: 1.2
Episode 500 | Reward: 3.8
Episode 1000 | Reward: 7.1
```

FPS is around 1800 on the RX 7900 XTX with 8 parallel envs. Not bad, but I know NVIDIA cards get more. ROCm overhead is real.

Had my first OOM crash — was using batch_size=512 with n_steps=2048 and 8 envs. The 24GB should have been enough but something weird with memory fragmentation. Dropped batch_size to 256 and it's fine now.

## 2026-06-01 — SAC implementation

Built the SAC agent from scratch. Tested on Pendulum-v1 first (quick sanity check), then tried my custom continuous CartPole.

Pendulum converges in about 20k steps — that's expected. The auto alpha tuning is nice, it settled around 0.15.

Key learning: **mixed precision makes a big difference on ROCm**. Got about 1.4x speedup with autocast. Some ops still fall back to fp32 but most of the matrix multiplications use fp16.

## 2026-06-05 — Atari wrapper issues

Frame stacking was broken — I was stacking in the wrong dimension order. Took me way too long to debug because the agent still "learned" something, it was just learning from garbage observations.

Fixed it, and now Breakout is actually getting good scores:
```
Episode 2000 | Reward: 24.3
Episode 5000 | Reward: 41.7
```

Also added the fire-on-reset logic which I forgot initially. Breakout literally can't start without pressing FIRE first. Classic noob mistake.

## 2026-06-08 — RLHF experiments

Started messing with RLHF. The reward model trains okay but the whole pipeline is unstable. The KL penalty is super sensitive:
- kl_coeff=0.01: policy ignores reward model
- kl_coeff=0.1: decent but slow learning
- kl_coeff=1.0: policy barely moves from reference

Going to try DPO next as an alternative. It's supposed to be simpler.

Also got hit with another OOM when I tried to increase the replay buffer to 2M. The buffer itself uses about 6GB at 1M entries with obs_dim=256. At 2M it was 12GB just for the buffer. Gotta be smarter about this.

## 2026-06-12 — Multi-agent training

Self-play is working! Set up a simple two-agent system where the main agent trains against a pool of past versions. The opponent pool strategy is key — training against just the latest version leads to weird strategies.

Tested with a simple game environment:
- Agent 1 vs latest opponent: wins 60%
- Agent 1 vs random opponent from pool: wins 75%
- Agent 1 vs oldest opponent from pool: wins 90%

This shows the agent is actually learning and improving over time.

Memory usage with multiple agents is around 8-10GB VRAM total. The 24GB card has plenty of headroom.

## 2026-06-17 — Cleanup and notes

Cleaned up the codebase. Things I want to do next:
- [ ] Try DDPG as another baseline
- [ ] Implement proper Atari benchmark (100M frames, standard games)
- [ ] Add WandB logging support
- [ ] Try multi-GPU with ROCm (need another AMD card first...)
- [ ] Port RLHF to actually work with language models
- [ ] Add reward shaping to custom envs

Performance summary so far:
- Breakout (PPO): ~45 avg reward after 10M steps
- Pendulum (SAC): -150 avg reward after 200k steps (good)
- Custom GridWorld: solves in ~50k steps with both PPO and SAC
- Self-play: working but needs more diverse environments

ROCm is honestly pretty good now. The main pain points are:
1. `torch.compile()` is unreliable
2. Some CUDA extensions don't work (need ROCm ports)
3. Memory management is slightly different (more fragmentation?)
4. Debugging is harder because error messages assume NVIDIA

But it works, and the RX 7900 XTX is genuinely good value for RL workloads with 24GB VRAM.
