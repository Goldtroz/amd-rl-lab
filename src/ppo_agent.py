"""
PPO (Proximal Policy Optimization) agent.

Built from scratch to run on AMD GPUs via ROCm.
This isn't the cleanest implementation but it works and it's fast enough.

ROCm notes:
- torch.cuda works the same as on NVIDIA (confusing but fine)
- Mixed precision with autocast works on ROCm 6.x
- torch.compile() is flaky — disabled by default, enable at your own risk
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Hyperparameters for PPO. Tweak these based on your env."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_steps: int = 2048
    batch_size: int = 64
    hidden_dim: int = 256
    use_mixed_precision: bool = True  # works on ROCm 6.x
    compile_model: bool = False  # don't trust this on ROCm yet


class ActorCritic(nn.Module):
    """
    Shared backbone actor-critic network.
    Supports both discrete and continuous action spaces.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        # shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, act_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            self.actor = nn.Linear(hidden_dim, act_dim)

        # critic head
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — helps with RL training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # smaller init for actor output
        if self.continuous:
            nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        value = self.critic(features)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)

        return dist, value

    def get_action_and_value(self, obs: torch.Tensor,
                              action: Optional[torch.Tensor] = None):
        dist, value = self.forward(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        if self.continuous:
            log_prob = log_prob.sum(-1)  # sum over action dims

        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(-1)

        return action, log_prob, entropy, value.squeeze(-1)


class PPOAgent:
    """
    PPO agent that actually works.

    Usage:
        agent = PPOAgent(obs_dim=4, act_dim=2)
        # collect rollouts, then:
        metrics = agent.update(rollouts)
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 config: Optional[PPOConfig] = None,
                 continuous: bool = False,
                 device: Optional[str] = None):
        self.config = config or PPOConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"PPO agent using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

        self.network = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=self.config.hidden_dim,
            continuous=continuous
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

        # optional: compile the model (ROCm support varies)
        if self.config.compile_model:
            try:
                self.network = torch.compile(self.network)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")

        self.scaler = torch.amp.GradScaler(enabled=self.config.use_mixed_precision)

    def select_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action for a single observation. Use during rollout collection."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(obs_t)

        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().item(),
            value.cpu().item()
        )

    def update(self, observations: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor) -> dict:
        """
        Run PPO update on collected rollout data.

        Returns dict of training metrics for logging.
        """
        batch_size = len(observations)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }

        # move data to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.config.n_epochs):
            # mini-batch updates
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, self.config.batch_size):
                end = start + self.config.batch_size
                mb_idx = indices[start:end]

                # mixed precision forward pass
                with torch.amp.autocast(device_type="cuda", enabled=self.config.use_mixed_precision):
                    _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                        observations[mb_idx], actions[mb_idx]
                    )

                    # ratio for clipping
                    log_ratio = new_log_probs - old_log_probs[mb_idx]
                    ratio = log_ratio.exp()

                    # approx kl divergence (for monitoring)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()

                    # clipped surrogate loss
                    mb_advantages = advantages[mb_idx]
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                        1 + self.config.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # value loss (clipped)
                    value_loss = nn.functional.mse_loss(new_values, returns[mb_idx])

                    # entropy bonus
                    entropy_loss = -entropy.mean()

                    # total loss
                    loss = (policy_loss
                            + self.config.value_coeff * value_loss
                            + self.config.entropy_coeff * entropy_loss)

                # backward pass with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # accumulate metrics
                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clip_frac"] += clip_frac.item()

        # average over all updates
        n_updates = self.config.n_epochs * max(1, batch_size // self.config.batch_size)
        for key in metrics:
            metrics[key] /= n_updates

        return metrics

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        last_gae = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded checkpoint from {path}")
