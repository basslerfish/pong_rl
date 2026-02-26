"""
Try to solve the Atari pong game with deep Q-learning.
We use a deep Q network (DQN) to predict state-action values given a state.
We train the network by calculating the loss between the predicted state-action value
and the bellman update (update = r + gamma * max(Q(s', a'))

Requirements for smooth learning (see Deepmind RL papers from 2013/2015).
1. Many wrappers are needed to convert atari images into useful learning material
2. We need a large buffer of experiences (transition records) to choose samples for SGD
3. We use epsilon-greedy with a decay to start with random actions, then slowly use more and more DQN-based actions
4. We keep a copy of our DQN that remains unchanged to predict Q(s', a') so that it is not affected by our weight updates.
    - This copy (target network) gets the newest weight only every x frames (where x is a large number)
"""
import time
from pathlib import Path

import ale_py  # to import atari games
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter

from pong_rl.wrappers import make_env
from pong_rl.model import DQN, calculate_loss
from pong_rl.data import ExperienceBuffer
from pong_rl.agent import Agent

# params
ENV_NAME = "PongNoFrameskip-v4"
OUTPUT_DIR = Path("/Users/mathis/Code/private_projects/learn_rl/results/lapan2024/q_learning")
LOG_DIR = Path("/Users/mathis/Code/private_projects/learn_rl/results/tb_runs/q_pong")
REWARD_LIMIT = 19  # we stop learning once model is that good
GAMMA = 0.99  # discount factor
BATCH_SIZE = 32
REPLAY_SIZE = 10_000  # buffer of transitions to learn from
LEARNING_RATE = 10 ** -4
SYNC_TARGET_FRAMES = 1_000  # how often to update the target network
REPLAY_START_SIZE = 10_000  # wait for experience before starting to learn

EPSILON_DECAY_FRAMES = 150_000
EPSILON_START = 1.0  # start with 100% chance of random action
EPSILON_END = 0.01  # end with almost no chance of random action

DEVICE = "cpu"


def main() -> None:
    print("---Setting up---")
    # set up device
    device = torch.device(DEVICE)

    # make environment
    env = make_env(ENV_NAME)
    print(f"Observation space: {env.observation_space.shape}")

    # make q models (CNNs that predict state-action values)
    net = DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    n_params = np.sum([p.numel() for p in net.parameters()])
    print(f"Parameters in network: {n_params:,.0f}")

    # create agent
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    # keep track of progress
    total_rewards = []
    best_m_reward = None
    frame_idx = 0
    last_time = time.time()
    log_dir = LOG_DIR / f"{time.time():.0f}"
    writer = SummaryWriter(log_dir=str(log_dir))
    last_loss = np.nan

    # training loop
    print("---Training---")
    while True:
        # compute random action chance
        epsilon_decay = EPSILON_START - frame_idx / EPSILON_DECAY_FRAMES
        epsilon = max(EPSILON_END, epsilon_decay)

        # take a step
        reward = agent.play_step(net, device, epsilon)

        # update weights if we have enough experiences to sample from
        if len(buffer) >= REPLAY_START_SIZE:
            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)  # sample experience buffer
            loss = calculate_loss(  # calculate loss between predicted q vals and bellman update
                batch=batch,
                net=net,
                tqt_net=tgt_net,
                device=device,
                gamma=GAMMA,
            )
            loss.backward()
            optimizer.step()
            last_loss = loss.item()

        # save & log
        if reward is not None:  # .play_step() only returns a reward when episode is done
            # save reward
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])  # running mean

            # save model
            if (best_m_reward is None) or (m_reward > best_m_reward):
                save_path = OUTPUT_DIR / f"best_{m_reward:.0f}.dat"
                torch.save(net.state_dict(), str(save_path))
                best_m_reward = m_reward

            # end training if our performance is good
            if m_reward > REWARD_LIMIT:
                print(f"Solved after {frame_idx + 1} frames!")
                break

            # log progress
            current_time = time.time()
            elapsed = current_time - last_time
            print(f"{frame_idx:,}: {len(total_rewards)} games, {reward:.0f} last reward, {m_reward:.3f} running mean, {elapsed:.3f}s game time")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("games", len(total_rewards), frame_idx)
            writer.add_scalar("buffer_length", len(buffer), frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("elapsed", elapsed, frame_idx)
            writer.add_scalar("running_mean_reward", m_reward, frame_idx)
            writer.add_scalar("last_loss", last_loss, frame_idx)
            last_time = current_time

        # give tgt network latest weights
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # update frame count
        frame_idx += 1


if __name__ == "__main__":
    main()


