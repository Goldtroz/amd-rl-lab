"""
Custom Gym/Gymnasium environments.

Mostly wrappers and a few custom envs for testing RL algorithms
without needing to set up Atari or MuJoCo every time.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple


class NoisyCartPole(gym.Wrapper):
    """
    CartPole but with observation noise. Makes it harder and tests
    whether your agent is robust to sensor noise.
    """

    def __init__(self, noise_std: float = 0.1, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        super().__init__(env)
        self.noise_std = noise_std

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info

    def _add_noise(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise


class ContinuousCartPole(gym.Wrapper):
    """
    CartPole with continuous action space.
    Maps [-1, 1] continuous to left/right with proportional force.
    """

    def __init__(self, **kwargs):
        env = gym.make("CartPole-v1", **kwargs)
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        # convert continuous to discrete
        discrete_action = 0 if action[0] < 0 else 1
        return self.env.step(discrete_action)


class GridWorld(gym.Env):
    """
    Simple grid world for quick RL experiments.
    Agent starts at (0,0), goal is at (size-1, size-1).
    Sparse reward: +1 at goal, -0.01 per step.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, size: int = 5, max_steps: int = 100,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # 4 actions: up, right, down, left
        self.action_space = spaces.Discrete(4)
        # observation: agent (x, y) normalized to [0, 1]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )

        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size - 1, size - 1])
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # movement mapping
        directions = {
            0: np.array([0, 1]),   # up
            1: np.array([1, 0]),   # right
            2: np.array([0, -1]),  # down
            3: np.array([-1, 0]),  # left
        }

        new_pos = self.agent_pos + directions[action]
        # clip to grid bounds
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos

        # check if reached goal
        reached_goal = np.array_equal(self.agent_pos, self.goal_pos)
        terminated = reached_goal
        truncated = self.steps >= self.max_steps

        reward = 1.0 if reached_goal else -0.01

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return self.agent_pos.astype(np.float32) / (self.size - 1)

    def render(self):
        if self.render_mode == "ansi":
            grid = [["." for _ in range(self.size)] for _ in range(self.size)]
            grid[self.goal_pos[1]][self.goal_pos[0]] = "G"
            grid[self.agent_pos[1]][self.agent_pos[0]] = "A"
            return "\n".join(" ".join(row) for row in reversed(grid))
        elif self.render_mode == "human":
            print(self.render())


class SparseRewardPendulum(gym.Wrapper):
    """
    Pendulum but with sparse reward instead of dense.
    Only gets reward when angle is near upright (within 0.2 rad).
    Much harder than normal Pendulum.
    """

    def __init__(self, threshold: float = 0.2, **kwargs):
        env = gym.make("Pendulum-v1", **kwargs)
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # extract angle from observation (cos(theta), sin(theta))
        cos_theta = obs[0]
        # sparse reward: 1 if near upright, 0 otherwise
        sparse_reward = 1.0 if cos_theta > np.cos(self.threshold) else 0.0
        return obs, sparse_reward, terminated, truncated, info


def make_env(env_id: str, **kwargs) -> gym.Env:
    """Factory function for creating environments."""
    custom_envs = {
        "noisy_cartpole": lambda: NoisyCartPole(**kwargs),
        "continuous_cartpole": lambda: ContinuousCartPole(**kwargs),
        "gridworld": lambda: GridWorld(**kwargs),
        "sparse_pendulum": lambda: SparseRewardPendulum(**kwargs),
    }

    if env_id in custom_envs:
        return custom_envs[env_id]()
    else:
        return gym.make(env_id, **kwargs)
