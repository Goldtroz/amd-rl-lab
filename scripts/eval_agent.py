"""
Evaluation script for trained agents.

Usage:
    python scripts/eval_agent.py --checkpoint checkpoints/ppo_breakout/ppo_final.pt
    python scripts/eval_agent.py --checkpoint checkpoints/sac_pendulum/sac_final.pt --episodes 100
"""

import argparse
import os
import sys
import logging
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ppo_agent import PPOAgent, PPOConfig
from src.sac_agent import SACAgent, SACConfig
from src.atari_wrapper import make_atari_env
from src.custom_env import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_ppo(checkpoint_path: str, env_name: str, n_episodes: int,
                 render: bool = False):
    """Evaluate a trained PPO agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", PPOConfig())

    # create environment
    if "ALE/" in env_name or "NoFrameskip" in env_name:
        env = make_atari_env(env_name, render_mode="human" if render else None)
    else:
        env = make_env(env_name, render_mode="human" if render else None)

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    # create agent and load weights
    agent = PPOAgent(obs_dim, act_dim, config=config, device=str(device))
    agent.network.load_state_dict(checkpoint["network"])

    # evaluate
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                time.sleep(0.02)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        logger.info(f"Episode {ep + 1}/{n_episodes} | Reward: {total_reward:.1f} | Length: {steps}")

    env.close()

    # summary
    rewards = np.array(episode_rewards)
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation over {n_episodes} episodes:")
    logger.info(f"  Mean reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    logger.info(f"  Min reward: {rewards.min():.1f}")
    logger.info(f"  Max reward: {rewards.max():.1f}")
    logger.info(f"  Mean length: {np.mean(episode_lengths):.1f}")
    logger.info(f"{'='*50}")

    return rewards.mean()


def evaluate_sac(checkpoint_path: str, env_name: str, n_episodes: int,
                 render: bool = False):
    """Evaluate a trained SAC agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # create environment
    env = make_env(env_name, render_mode="human" if render else None)
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape[0]

    # create agent and load weights
    agent = SACAgent(obs_dim, act_dim, device=str(device))
    agent.actor.load_state_dict(checkpoint["actor"])

    # evaluate
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if render:
                time.sleep(0.02)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        logger.info(f"Episode {ep + 1}/{n_episodes} | Reward: {total_reward:.1f}")

    env.close()

    rewards = np.array(episode_rewards)
    logger.info(f"\n{'='*50}")
    logger.info(f"SAC Evaluation over {n_episodes} episodes:")
    logger.info(f"  Mean reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    logger.info(f"  Min: {rewards.min():.1f} | Max: {rewards.max():.1f}")
    logger.info(f"{'='*50}")

    return rewards.mean()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5",
                        help="Environment name")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                        help="Algorithm type")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    args = parser.parse_args()

    if args.algo == "ppo":
        evaluate_ppo(args.checkpoint, args.env, args.episodes, args.render)
    elif args.algo == "sac":
        evaluate_sac(args.checkpoint, args.env, args.episodes, args.render)


if __name__ == "__main__":
    main()
