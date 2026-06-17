"""
Multi-agent training utilities.

Supports:
- Self-play (same policy playing against itself)
- Independent learners (each agent has own policy)
- Simple communication protocols (WIP)

Running multiple agents on a single RX 7900 XTX is fine as long
as the environments aren't too complex. The 24GB VRAM helps a lot.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
from copy import deepcopy

from .ppo_agent import PPOAgent, PPOConfig, ActorCritic

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentConfig:
    """Config for multi-agent training."""
    n_agents: int = 2
    shared_policy: bool = True  # self-play vs independent
    self_play: bool = True
    opponent_update_freq: int = 100  # episodes between opponent updates
    population_size: int = 5  # for population-based training


class SelfPlayTrainer:
    """
    Self-play training for two-player games.

    The main agent trains against a pool of past versions of itself.
    This prevents overfitting to a single opponent strategy.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 config: Optional[MultiAgentConfig] = None,
                 ppo_config: Optional[PPOConfig] = None,
                 device: Optional[str] = None):
        self.config = config or MultiAgentConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # main agent (the one we're training)
        self.agent = PPOAgent(obs_dim, act_dim, config=ppo_config, device=device)

        # opponent pool (past versions of the main agent)
        self.opponent_pool: List[ActorCritic] = []
        self._save_to_pool()

        self.episode_count = 0
        logger.info(f"Self-play trainer initialized with {self.config.n_agents} agents")

    def _save_to_pool(self):
        """Save current agent to opponent pool."""
        opponent = deepcopy(self.agent.network)
        opponent.eval()
        self.opponent_pool.append(opponent)

        # keep pool size bounded
        if len(self.opponent_pool) > self.config.population_size:
            self.opponent_pool.pop(0)

    def select_opponent_action(self, obs: np.ndarray,
                                opponent_idx: Optional[int] = None) -> np.ndarray:
        """Select action using a random opponent from the pool."""
        if opponent_idx is None:
            opponent_idx = np.random.randint(len(self.opponent_pool))

        opponent = self.opponent_pool[opponent_idx]

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            dist, _ = opponent(obs_t)
            action = dist.sample()

        return action.cpu().numpy().flatten()

    def select_agent_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action for the main agent."""
        return self.agent.select_action(obs)

    def maybe_update_opponent(self):
        """Update opponent pool if enough episodes have passed."""
        self.episode_count += 1

        if (self.config.self_play and
                self.episode_count % self.config.opponent_update_freq == 0):
            self._save_to_pool()
            logger.info(f"Opponent pool updated. Pool size: {len(self.opponent_pool)}")

    def get_opponent_pool_size(self) -> int:
        return len(self.opponent_pool)


class IndependentLearners:
    """
    Multiple independent agents, each with their own policy.
    Good for cooperative or competitive multi-agent settings
    where agents don't share a policy.
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int,
                 ppo_config: Optional[PPOConfig] = None,
                 device: Optional[str] = None):
        self.n_agents = n_agents
        self.agents = [
            PPOAgent(obs_dim, act_dim, config=ppo_config, device=device)
            for _ in range(n_agents)
        ]
        logger.info(f"Independent learners: {n_agents} agents")

    def select_actions(self, observations: List[np.ndarray]) -> List[Tuple]:
        """Select actions for all agents in parallel."""
        return [
            agent.select_action(obs)
            for agent, obs in zip(self.agents, observations)
        ]

    def update_all(self, rollouts: List[Dict]) -> List[Dict]:
        """Update all agents with their respective rollout data."""
        return [
            agent.update(**rollout)
            for agent, rollout in zip(self.agents, rollouts)
        ]

    def save_all(self, path_prefix: str):
        """Save all agent checkpoints."""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent_{i}.pt")

    def load_all(self, path_prefix: str):
        """Load all agent checkpoints."""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent_{i}.pt")


class SimpleMultiAgentEnv:
    """
    Simple multi-agent environment wrapper.
    Wraps a single-agent env into a multi-agent one by duplicating
    the environment and running agents in parallel.

    For proper multi-agent envs, use PettingZoo.
    """

    def __init__(self, env_fn, n_agents: int):
        self.envs = [env_fn() for _ in range(n_agents)]
        self.n_agents = n_agents

    def reset(self) -> List[np.ndarray]:
        """Reset all environments."""
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        return observations

    def step(self, actions: List) -> Tuple[List, List, List, List]:
        """Step all environments."""
        observations, rewards, dones, truncateds = [], [], [], []

        for env, action in zip(self.envs, actions):
            obs, reward, done, trunc, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(trunc)

        return observations, rewards, dones, truncateds

    def close(self):
        for env in self.envs:
            env.close()
