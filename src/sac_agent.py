"""
SAC (Soft Actor-Critic) implementation.

Continuous action spaces only. Works well on MuJoCo / custom envs.
Another one that runs fine on AMD via ROCm.

Key difference from PPO: off-policy, learns from replay buffer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class SACConfig:
    """SAC hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None  # auto-set to -act_dim if None
    hidden_dim: int = 256
    buffer_size: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 1000
    update_every: int = 1
    updates_per_step: int = 1
    use_mixed_precision: bool = True


class ReplayBuffer:
    """Simple replay buffer. Could be more efficient but this works."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.observations[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]).unsqueeze(-1),
            torch.FloatTensor(self.next_observations[indices]),
            torch.FloatTensor(self.dones[indices]).unsqueeze(-1),
        )


class SquashedGaussianActor(nn.Module):
    """Actor that outputs squashed Gaussian actions in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, obs: torch.Tensor):
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # log prob with tanh squashing correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mean


class Critic(nn.Module):
    """Twin Q-networks for SAC (clipped double Q-learning)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    """
    SAC agent for continuous action spaces.

    Usage:
        agent = SACAgent(obs_dim=17, act_dim=6)
        # in training loop:
        action = agent.select_action(obs)
        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.update()
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 config: Optional[SACConfig] = None,
                 device: Optional[str] = None):
        self.config = config or SACConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"SAC agent using device: {self.device}")

        # networks
        self.actor = SquashedGaussianActor(obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, self.config.hidden_dim).to(self.device)

        # copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr)

        # auto alpha tuning
        if self.config.auto_alpha:
            target_entropy = self.config.target_entropy or -act_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr)
            self.target_entropy = target_entropy
        else:
            self.log_alpha = torch.log(torch.tensor(self.config.alpha)).to(self.device)

        # replay buffer
        self.buffer = ReplayBuffer(self.config.buffer_size, obs_dim, act_dim)
        self.total_steps = 0

        # scaler for mixed precision
        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action. Deterministic for eval, stochastic for training."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if deterministic:
                mean, _ = self.actor(obs_t)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        """Run one SAC update step. Call after warmup."""
        if self.total_steps < self.config.warmup_steps:
            return {}

        if self.total_steps % self.config.update_every != 0:
            return {}

        all_metrics = {}

        for _ in range(self.config.updates_per_step):
            # sample batch
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_obs = next_obs.to(self.device)
            dones = dones.to(self.device)

            with torch.amp.autocast(device_type="cuda", enabled=self.config.use_mixed_precision):
                # ---- critic update ----
                with torch.no_grad():
                    next_actions, next_log_probs, _ = self.actor.sample(next_obs)
                    q1_target, q2_target = self.critic_target(next_obs, next_actions)
                    q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
                    target_q = rewards + self.config.gamma * (1 - dones) * q_target

                q1, q2 = self.critic(obs, actions)
                critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()

            # ---- actor update ----
            with torch.amp.autocast(device_type="cuda", enabled=self.config.use_mixed_precision):
                new_actions, log_probs, _ = self.actor.sample(obs)
                q1_new, q2_new = self.critic(obs, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            # ---- alpha update ----
            if self.config.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # soft update target networks
            self._soft_update()

            all_metrics = {
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha": self.alpha.item(),
                "buffer_size": self.buffer.size,
            }

        return all_metrics

    def _soft_update(self):
        """Polyak averaging for target network."""
        for param, target_param in zip(self.critic.parameters(),
                                        self.critic_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "total_steps": self.total_steps,
        }, path)
        logger.info(f"SAC checkpoint saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha = checkpoint["log_alpha"]
        self.total_steps = checkpoint["total_steps"]
        logger.info(f"SAC checkpoint loaded from {path}")
