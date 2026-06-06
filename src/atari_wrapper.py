"""
Atari environment wrapper.

Handles frame preprocessing, stacking, and action repeat.
Based on the standard DeepMind Atari wrapper but simplified.
Works with Gymnasium's Atari environments.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Optional, Tuple
import cv2


class AtariPreprocessing(gym.Wrapper):
    """
    Standard Atari preprocessing:
    - Grayscale
    - Resize to 84x84
    - Frame stacking (4 frames)
    - Action repeat (frame skip)
    - Episode life (life loss = episode end)
    - Fire on reset (press FIRE to start)
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4,
                 stack_frames: int = 4, img_size: int = 84,
                 terminal_on_life_loss: bool = True,
                 fire_on_reset: bool = True):
        super().__init__(env)

        self.frame_skip = frame_skip
        self.stack_frames = stack_frames
        self.img_size = img_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.fire_on_reset = fire_on_reset

        self.frames = deque(maxlen=stack_frames)

        # override observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(stack_frames, img_size, img_size),
            dtype=np.uint8
        )

        self.lives = 0
        self.was_real_done = True

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # just lost a life, don't fully reset
            obs, _, _, _, info = self.env.step(0)  # no-op

        if self.fire_on_reset:
            # press FIRE to start the game
            obs, _, _, _, _ = self.env.step(1)

        self.lives = self._get_lives()
        self._init_frames(obs)
        return self._get_obs(), info

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        self.lives = self._get_lives()

        # check for life loss
        if self.terminal_on_life_loss:
            if self.lives < self._get_lives():
                # life lost — treat as episode end (but not true done)
                terminated = True

        self.was_real_done = terminated or truncated
        self._update_frames(obs)

        return self._get_obs(), total_reward, terminated, truncated, info

    def _get_lives(self):
        """Get remaining lives. Returns 0 if not available."""
        if hasattr(self.env, "ale"):
            return self.env.ale.lives()
        return 0

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and resize."""
        gray = np.mean(frame, axis=2).astype(np.uint8)
        resized = cv2.resize(gray, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_AREA)
        return resized

    def _init_frames(self, obs: np.ndarray):
        """Fill frame buffer with initial observation."""
        processed = self._preprocess_frame(obs)
        for _ in range(self.stack_frames):
            self.frames.append(processed)

    def _update_frames(self, obs: np.ndarray):
        """Add new frame to buffer."""
        processed = self._preprocess_frame(obs)
        self.frames.append(processed)

    def _get_obs(self) -> np.ndarray:
        """Stack frames into single observation."""
        return np.stack(list(self.frames), axis=0)


class RewardScaler(gym.Wrapper):
    """Scale rewards by a constant. Useful for normalizing across games."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward * self.scale, terminated, truncated, info


class ClipReward(gym.Wrapper):
    """Clip rewards to {-1, 0, 1}."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


def make_atari_env(env_name: str = "ALE/Breakout-v5",
                   frame_skip: int = 4,
                   stack_frames: int = 4,
                   clip_reward: bool = True,
                   render_mode: Optional[str] = None) -> gym.Env:
    """
    Create a fully wrapped Atari environment.

    Args:
        env_name: Atari game name (e.g., "ALE/Breakout-v5")
        frame_skip: Number of frames to repeat each action
        stack_frames: Number of frames to stack
        clip_reward: Whether to clip rewards to {-1, 0, 1}
        render_mode: "human" for visualization, None for headless

    Returns:
        Wrapped Gymnasium environment
    """
    env = gym.make(env_name, render_mode=render_mode,
                   frameskip=1)  # we handle frame skip ourselves

    env = AtariPreprocessing(
        env,
        frame_skip=frame_skip,
        stack_frames=stack_frames,
    )

    if clip_reward:
        env = ClipReward(env)

    return env


# quick test
if __name__ == "__main__":
    env = make_atari_env("ALE/Breakout-v5")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")  # (4, 84, 84)
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    total_reward = 0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
            print(f"Episode done, reward: {total_reward}")
            total_reward = 0
