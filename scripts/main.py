"""
Use a pre-trained DQN to play pong.
--file: Can use path to weights file as input argument.
"""
import argparse
from pathlib import Path

import ale_py  # needed to load atari games
import torch

from pong_rl.wrappers import make_env
from pong_rl.agent import Agent
from pong_rl.model import DQN
from pong_rl.data import ExperienceBuffer


# params
ENV_NAME = "PongNoFrameskip-v4"


def main() -> None:
    # argument parsing
    default_file = Path.cwd().parent / "data" / "best_weights.dat"
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to model weights file.", default=str(default_file))
    args = parser.parse_args()

    # load environment
    env = make_env(ENV_NAME, render_mode="human")

    # instantiate model
    device = torch.device("cpu")  # small enough to run on cpu
    net = DQN(env.observation_space.shape, env.action_space.n)

    # load weights
    weights_file = Path(args.file)
    assert weights_file.is_file(), f"Weights file {weights_file} does not exist."
    state_dict = torch.load(str(weights_file), map_location=device)
    net.load_state_dict(state_dict)

    # prep agent
    buffer = ExperienceBuffer(0)  # we don't need to save experiences now
    agent = Agent(env, buffer)

    # play episode and render
    episode_reward = None
    while episode_reward is None:
        episode_reward = agent.play_step(net, device=device, epsilon=0, render=True)
    print(f"Episode completed - reward: {episode_reward:.0f}")


if __name__ == "__main__":
    main()
