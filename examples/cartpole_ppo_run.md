# CartPole PPO Training Run

## Setup

- Environment: CartPole-v1
- Algorithm: PPO
- Steps: 50,000
- Learning rate: 3e-4
- Gamma: 0.99
- Clip: 0.2
- Epochs per update: 10

## Training curve

```
Episode  Reward
1        22
100      45
500      120
1000     195
1500     250
2000     300 (solved)
```

CartPole-v1 is 'solved' at 195 average over 100 episodes.

## Observations

1. Learning rate matters a lot: 1e-3 diverged, 1e-4 was too slow, 3e-4 was the sweet spot
2. Clip value: 0.2 worked well. 0.1 was too conservative, 0.3 was too aggressive
3. Advantage normalization helped stability significantly
4. Entropy bonus: small value (0.01) helped exploration

## Hyperparameter sensitivity

| Parameter | Value | Result |
|-----------|-------|--------|
| lr=1e-3 | Diverged | Bad |
| lr=3e-4 | Solved at 2000 steps | Good |
| lr=1e-4 | Slow convergence | Okay |
| clip=0.1 | Slow but stable | Okay |
| clip=0.2 | Fast convergence | Good |
| clip=0.3 | Unstable | Bad |
