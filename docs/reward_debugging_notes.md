# Reward Debugging Notes

## Common problems

### 1. Reward hacking
Agent finds a loophole in the reward function.

Example: In a navigation task, agent learns to spin in circles because each step gives +0.1 for 'moving'. The agent maximizes steps, not progress.

Fix: Redesign reward to penalize spinning or reward progress toward goal.

### 2. Sparse rewards
Agent gets reward only at the end of the episode.

Example: Chess. You only get +1 for winning, -1 for losing. No signal during the game.

Fix: Add intermediate rewards (piece advantage, center control).

### 3. Reward scale
Rewards are too large or too small.

Example: Reward of 10000 per step. The value function explodes.

Fix: Normalize rewards to [-1, 1] range.

### 4. Conflicting rewards
Different reward components send conflicting signals.

Example: Reward for speed + reward for safety. Agent is confused.

Fix: Weight the components carefully. Test each component separately.

## Debugging tools

1. **Print reward components**: See which part of the reward is dominant
2. **Plot reward curves**: Look for sudden drops or plateaus
3. **Compare with random agent**: If your agent is worse than random, something is wrong
4. **Check state normalization**: Unnormalized states can cause training instability

## Reward shaping tips

1. Start with the simplest reward that captures the goal
2. Add shaping rewards gradually
3. Test each shaping component individually
4. Use a baseline (random agent) to verify reward signal is meaningful
5. Monitor the value function: if it diverges, the reward scale is wrong
