# Simple Trading Environment

## Design

A minimal trading environment for testing RL agents.

- State: [price_change_5d, price_change_20d, position, balance]
- Actions: buy, sell, hold
- Reward: portfolio value change
- Episode: 252 trading days (1 year)

## Implementation

```python
class SimpleTradingEnv:
    def __init__(self, prices):
        self.prices = prices
        self.position = 0
        self.balance = 10000
        self.day = 0
    
    def step(self, action):
        # 0=hold, 1=buy, 2=sell
        if action == 1 and self.balance >= self.prices[self.day]:
            self.position += 1
            self.balance -= self.prices[self.day]
        elif action == 2 and self.position > 0:
            self.position -= 1
            self.balance += self.prices[self.day]
        
        self.day += 1
        portfolio_value = self.balance + self.position * self.prices[self.day]
        reward = portfolio_value - 10000  # Change from initial
        done = self.day >= len(self.prices) - 1
        
        return self._get_state(), reward, done
```

## PPO results

- After 100K steps: agent learned to buy low, sell high (on training data)
- On test data: performance degraded significantly (overfitting)
- Lesson: RL agents overfit to training environments just like supervised models

## What I'm testing next

- Regularization to prevent overfitting
- Different reward functions (Sharpe ratio, max drawdown)
- Longer episodes with more price history
