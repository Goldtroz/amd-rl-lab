"""
Basic RLHF (Reinforcement Learning from Human Feedback) trainer.

This is experimental / WIP. Still figuring out the best approach.
Currently implements a simple reward model + PPO fine-tuning loop.

Not production quality, more of a learning exercise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Config for RLHF training."""
    reward_model_lr: float = 1e-4
    policy_lr: float = 1e-5
    reward_model_hidden: int = 256
    reward_model_epochs: int = 10
    kl_coeff: float = 0.1
    clip_epsilon: float = 0.2
    batch_size: int = 32
    max_seq_len: int = 512


class RewardModel(nn.Module):
    """
    Reward model that learns from human preferences.

    Takes an observation/trajectory and predicts a scalar reward.
    In the simplest case, we train on pairwise comparisons:
    "trajectory A is better than trajectory B"
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreferenceDataset:
    """
    Stores human preference data.
    Each entry is a pair of trajectories with a label
    indicating which one the human preferred.
    """

    def __init__(self, max_size: int = 10000):
        self.trajectories_a = []
        self.trajectories_b = []
        self.preferences = []  # 0 = A preferred, 1 = B preferred
        self.max_size = max_size

    def add_preference(self, traj_a: np.ndarray, traj_b: np.ndarray,
                       preference: int):
        """
        Add a human preference.

        Args:
            traj_a: First trajectory (obs, action, reward sequences)
            traj_b: Second trajectory
            preference: 0 if A is preferred, 1 if B is preferred
        """
        self.trajectories_a.append(traj_a)
        self.trajectories_b.append(traj_b)
        self.preferences.append(preference)

        # trim if too big
        if len(self.preferences) > self.max_size:
            self.trajectories_a = self.trajectories_a[-self.max_size:]
            self.trajectories_b = self.trajectories_b[-self.max_size:]
            self.preferences = self.preferences[-self.max_size:]

    def sample_batch(self, batch_size: int) -> Tuple:
        """Sample a batch of preference pairs."""
        n = len(self.preferences)
        if n < batch_size:
            batch_size = n

        indices = np.random.choice(n, size=batch_size, replace=False)

        traj_a = np.array([self.trajectories_a[i] for i in indices])
        traj_b = np.array([self.trajectories_b[i] for i in indices])
        prefs = np.array([self.preferences[i] for i in indices])

        return (
            torch.FloatTensor(traj_a),
            torch.FloatTensor(traj_b),
            torch.FloatTensor(prefs),
        )


class RLHFTrainer:
    """
    Basic RLHF training loop.

    1. Collect trajectories from current policy
    2. Get human preferences on trajectory pairs
    3. Train reward model on preferences
    4. Fine-tune policy with reward model + KL penalty

    TODO: Add automated preference collection
    TODO: Try DPO (Direct Preference Optimization) instead
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 config: Optional[RLHFConfig] = None,
                 device: Optional[str] = None):
        self.config = config or RLHFConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # reward model
        self.reward_model = RewardModel(
            input_dim=obs_dim,
            hidden_dim=self.config.reward_model_hidden
        ).to(self.device)

        self.reward_optimizer = optim.Adam(
            self.reward_model.parameters(),
            lr=self.config.reward_model_lr
        )

        # preference dataset
        self.preference_data = PreferenceDataset()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        logger.info(f"RLHF trainer initialized on {self.device}")

    def add_human_preference(self, traj_a: np.ndarray, traj_b: np.ndarray,
                              preferred: str = "a"):
        """
        Add a human preference between two trajectories.

        Args:
            traj_a: First trajectory observations
            traj_b: Second trajectory observations
            preferred: "a" or "b" — which trajectory the human preferred
        """
        label = 0 if preferred == "a" else 1
        self.preference_data.add_preference(traj_a, traj_b, label)

    def train_reward_model(self, n_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train the reward model on collected human preferences.

        Uses Bradley-Terry model: P(A > B) = sigmoid(R(A) - R(B))
        """
        if len(self.preference_data.preferences) < 10:
            logger.warning("Not enough preference data to train (need >= 10)")
            return {}

        n_epochs = n_epochs or self.config.reward_model_epochs
        total_loss = 0.0

        for epoch in range(n_epochs):
            traj_a, traj_b, prefs = self.preference_data.sample_batch(
                self.config.batch_size
            )
            traj_a = traj_a.to(self.device)
            traj_b = traj_b.to(self.device)
            prefs = prefs.to(self.device)

            # compute rewards for each trajectory
            reward_a = self.reward_model(traj_a.mean(dim=1))  # simple pooling
            reward_b = self.reward_model(traj_b.mean(dim=1))

            # Bradley-Terry loss
            # P(A preferred) = sigmoid(R(A) - R(B))
            reward_diff = reward_a - reward_b
            # prefs: 0 = A preferred, 1 = B preferred
            # so target is (1 - prefs) for A being preferred
            target = 1.0 - prefs
            loss = nn.functional.binary_cross_entropy_with_logits(
                reward_diff.squeeze(), target
            )

            self.reward_optimizer.zero_grad()
            loss.backward()
            self.reward_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_epochs
        logger.info(f"Reward model trained — avg loss: {avg_loss:.4f}")
        return {"reward_model_loss": avg_loss}

    def compute_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute reward from the learned reward model."""
        with torch.no_grad():
            return self.reward_model(observations.to(self.device))

    def compute_kl_penalty(self, current_policy_logprobs: torch.Tensor,
                            reference_policy_logprobs: torch.Tensor) -> torch.Tensor:
        """
        KL divergence penalty to keep the policy close to the reference.
        This prevents reward hacking.
        """
        kl = current_policy_logprobs - reference_policy_logprobs
        return self.config.kl_coeff * kl.mean()

    def get_reward_model_state(self) -> Dict:
        """Save reward model state for checkpointing."""
        return {
            "reward_model": self.reward_model.state_dict(),
            "reward_optimizer": self.reward_optimizer.state_dict(),
            "n_preferences": len(self.preference_data.preferences),
        }

    def load_reward_model_state(self, state: Dict):
        """Load reward model from checkpoint."""
        self.reward_model.load_state_dict(state["reward_model"])
        self.reward_optimizer.load_state_dict(state["reward_optimizer"])
        logger.info(f"Loaded reward model with {state['n_preferences']} preferences")
