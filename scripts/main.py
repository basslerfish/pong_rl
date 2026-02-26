"""
Use a DQN to play pong.
"""
import ale_py  # needed to load atari games
import gymnasium as gym

from pong_rl.wrappers import make_env


# params
ENV_NAME = "PongNoFrameskip-v4"

# load model

# load environment
env = make_env(ENV_NAME)

# play and display