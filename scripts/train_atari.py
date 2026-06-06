"""
Training script for Atari environments.

Usage:
    python scripts/train_atari.py --config configs/ppo_atari.yaml
    python scripts/train_atari.py --env ALE/Pong-v5 --timesteps 5000000
"""

import argparse
import yaml
import os
import sys
import logging
import time
import numpy as np
import torch
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppo_agent import PPOAgent, PPOConfig
from src.atari_wrapper import make_atari_env
from src.custom_env import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_env(config: dict):
    """Create environment from config."""
    env_name = config["env"]["name"]

    # check if it's an Atari game
    if "ALE/" in env_name or "NoFrameskip" in env_name:
        return make_atari_env(
            env_name=env_name,
            frame_skip=config["env"].get("frame_skip", 4),
            stack_frames=config["env"].get("stack_frames", 4),
            clip_reward=config["env"].get("clip_reward", True),
        )
    else:
        return make_env(env_name)


def train(config: dict):
    """Main training loop."""
    total_timesteps = config["training"]["total_timesteps"]
    log_dir = config["training"]["log_dir"]
    checkpoint_dir = config["training"]["checkpoint_dir"]
    save_freq = config["training"]["save_freq"]

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create environment
    env = create_env(config)
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    logger.info(f"Environment: {config['env']['name']}")
    logger.info(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    # create agent
    ppo_config = PPOConfig(**config["agent"])
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        config=ppo_config,
    )

    # training loop
    logger.info(f"Starting training for {total_timesteps:,} timesteps")
    start_time = time.time()

    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    timestep = 0

    # rollout buffers
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    while timestep < total_timesteps:
        # collect rollout
        for step in range(ppo_config.n_steps):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            obs = next_obs
            episode_reward += reward
            timestep += 1

            if done:
                episode_count += 1
                logger.info(f"Episode {episode_count} | Reward: {episode_reward:.1f} | Timestep: {timestep:,}")
                episode_reward = 0
                obs, _ = env.reset()

        # compute advantages and returns
        obs_tensor = torch.FloatTensor(np.array(observations))
        action_tensor = torch.FloatTensor(np.array(actions))
        log_prob_tensor = torch.FloatTensor(np.array(log_probs))
        reward_array = np.array(rewards)
        done_array = np.array(dones)
        value_array = np.array(values)

        advantages, returns = agent.compute_gae(reward_array, value_array, done_array)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # PPO update
        metrics = agent.update(obs_tensor, action_tensor, log_prob_tensor,
                               returns_tensor, advantages_tensor)

        # log metrics
        elapsed = time.time() - start_time
        fps = timestep / elapsed
        logger.info(
            f"Step {timestep:,}/{total_timesteps:,} | "
            f"FPS: {fps:.0f} | "
            f"Policy loss: {metrics['policy_loss']:.4f} | "
            f"Value loss: {metrics['value_loss']:.4f} | "
            f"Entropy: {metrics['entropy']:.4f} | "
            f"KL: {metrics['approx_kl']:.4f}"
        )

        # clear rollout buffers
        observations.clear()
        actions.clear()
        log_probs.clear()
        rewards.clear()
        dones.clear()
        values.clear()

        # save checkpoint
        if timestep >= save_freq and timestep % save_freq < ppo_config.n_steps:
            checkpoint_path = os.path.join(checkpoint_dir, f"ppo_step_{timestep}.pt")
            agent.save(checkpoint_path)

    # final save
    final_path = os.path.join(checkpoint_dir, "ppo_final.pt")
    agent.save(final_path)
    logger.info(f"Training complete! Final checkpoint: {final_path}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Atari")
    parser.add_argument("--config", type=str, default="configs/ppo_atari.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--env", type=str, default=None,
                        help="Override environment name")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total timesteps")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.env:
        config["env"]["name"] = args.env
    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps

    train(config)


if __name__ == "__main__":
    main()
