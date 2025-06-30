# ROCm Notes — AMD RL Lab

## Test Target

- **ROCm version:** 6.x (6.2 preferred)
- **PyTorch:** `torch` with ROCm backend (`--index-url https://download.pytorch.org/whl/rocm6.2`)
- **GPU strategy:** Single GPU first, multi-GPU env vectorization later
- **Primary card:** RX 7900 XTX (24GB VRAM)

## Current Blockers

- `torch.compile()` still unstable for RL training loops with dynamic control flow (episode boundaries cause recompilation)
- Mixed precision autocast needs careful handling in PPO advantage estimation — bf16 causes subtle NaN issues in GAE computation

## Planned Tests

| Test | Metric | Status |
|------|--------|--------|
| fp16 training (PPO) | Steps/sec, final reward | Pending |
| bf16 training (PPO) | Steps/sec, final reward | Pending |
| fp16 training (SAC) | Q-value stability, reward | Pending |
| Batch size scaling | VRAM usage vs. throughput | Pending |
| GPU utilization % | During rollout collection vs. update | Pending |
| Memory peak | Max VRAM during gradient update | Pending |

## Repo-Specific Notes

### PPO Training Loop

- The rollout collection phase is CPU-bound (env stepping) — GPU utilization drops to ~15% during this phase
- The policy/value update phase hits ~85% GPU utilization with 2048-step rollouts
- Key optimization: use larger `num_envs` to keep the GPU fed during update phases
- `torch.no_grad()` context during rollout is critical for VRAM — without it, the computation graph leaks memory

### SAC Training Loop

- Critic target network soft updates are cheap on GPU — negligible overhead
- Replay buffer sampling is the bottleneck — moving to GPU-side buffer if memory allows
- Entropy tuning (auto-alpha) can cause instability on fp16; recommend bf16 or fp32 for alpha parameter

### GPU Utilization Monitoring

```bash
# Real-time GPU monitoring during training
watch -n 1 rocm-smi

# Or in Python
import subprocess
subprocess.run(["rocm-smi", "--showuse"])
```

### Environment Step Throughput

- Atari envs: ~4000 steps/sec single env on CPU
- With vectorized envs (8 parallel): ~28,000 steps/sec on CPU
- GPU advantage computation: ~0.3ms per batch of 2048 steps (fp32), ~0.15ms (fp16)
- The sweet spot is 16+ vectorized envs to saturate the GPU during PPO updates

### Known ROCm Quirks

- `HSA_OVERRIDE_GFX_VERSION` not needed on RDNA3 (Navi 31) — it auto-detects
- Memory allocation is slightly more aggressive than CUDA — use `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for long training runs
- TensorBoard logging with `torch.utils.tensorboard` works fine on ROCm
