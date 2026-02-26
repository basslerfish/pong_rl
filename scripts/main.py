"""
Use a pre-trained DQN to play pong.
"""
import ale_py  # needed to load atari games
import gymnasium as gym

from pong_rl.wrappers import make_env
from pong_rl.agent import Agent
from pong_rl.model import DQN


# params
ENV_NAME = "PongNoFrameskip-v4"

# load environment
env = make_env(ENV_NAME)

# load model
net = DQN()



# play and display