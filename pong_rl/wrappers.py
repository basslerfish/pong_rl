"""
Wrappers for Atari games.

It is a long way from the raw frames of an Atari game to ML-usable input.
Fortunately, stable_baselines3 handles most of the data preprocessing.
"""
import collections

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.atari_wrappers import AtariWrapper


class ImageToPyTorch(gym.ObservationWrapper):
    """Change image channel axis to position 0."""
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs = self.observation_space
        assert isinstance(obs, gym.spaces.Box)
        assert len(obs.shape) == 3
        new_shape = (obs.shape[-1], obs.shape[0], obs.shape[1])
        self.observation_space = spaces.Box(
            low=obs.low.min(),
            high=obs.high.max(),
            shape=new_shape,
            dtype=obs.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Move channel axis to front.
        That's how PyTorch expects image data.
        """
        observation = np.moveaxis(observation, 2, 0)
        return observation


class BufferWrapper(gym.ObservationWrapper):
    """
    Create a frame buffer to stack multiple frames into one state.
    This way, movement information can be extracted (which would not be possible from a single still image).
    """
    def __init__(self, env: gym.Env, n_steps: int) -> None:
        super().__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = spaces.Box(
            obs.low.repeat(n_steps, axis=0),
            obs.high.repeat(n_steps, axis=0),
            dtype=obs.dtype,
        )
        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Return stack of frames in buffer (instead of a single frame).
        """
        self.buffer.append(observation)
        out = np.concatenate(self.buffer)
        return out

    def reset(
            self, *,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple:
        """When resetting, fill the buffer with empty frames."""
        for _ in range(self.buffer.maxlen - 1):
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset()
        return self.observation(obs), extra


def make_env(env_name: str, **kwargs) -> gym.Env:
    """
    Make an environment and apply all wrappers.
    """
    env = gym.make(env_name, **kwargs)
    env = AtariWrapper(env, clip_reward=False, noop_max=0)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env
