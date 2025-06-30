# Expected Reward Curve — PPO on Breakout

## Training Progression

When running the example config (`ppo_atari_config.yaml`) on Breakout, expect the following reward curve:

### Phase 1: Random Policy (0-500K steps)
- Average reward: ~1-5 points
- Agent learns basic input mapping, occasional paddle hits
- High entropy — policy is still exploratory

### Phase 2: Early Learning (500K-2M steps)
- Average reward climbs from ~5 to ~50 points
- Agent learns to hit the ball somewhat consistently
- Entropy drops as policy becomes more deterministic

### Phase 3: Competent Play (2M-5M steps)
- Average reward: ~50-200 points
- Agent clears rows and learns brick patterns
- Occasional high-scoring episodes (~400+ points)

### Phase 4: Mastery (5M-10M steps)
- Average reward: ~200-400 points
- Consistent play with few deaths
- Some runs can reach 500+ points

## Visual Shape

```
Reward
  |
400|                                          ___________
   |                                    _____/
300|                               ____/
   |                          ____/
200|                     ____/
   |                ____/
100|           ____/
   |      ____/
  5|_____/_______________________________________________
   0     1M    2M    3M    4M    5M    6M    7M    8M    9M   10M
                        Training Steps
```

## Key Observations

- **Wall clock on CPU:** ~18 hours for 10M steps
- **Expected on RX 7900 XTX:** ~3-4 hours (GPU acceleration on policy updates)
- The bottleneck on GPU is still env stepping (CPU-bound), not network updates
- Mixed precision (fp16) saves ~30% VRAM, allowing larger batch sizes or more parallel envs
