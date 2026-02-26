"""
Try to solve the Atari pong game with deep Q-learning.
Pass the following arguments:
--dev: device, cpu or gpu
--logdir: where to save tensorboard logs
--savedir: where to save model weights.

How it works:
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
import argparse
import time
from pathlib import Path

import ale_py  # to import atari games
import pandas as pd
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
REWARD_LIMIT = 19  # we stop learning once model is that good
GAMMA = 0.99  # discount factor
BATCH_SIZE = 32  # experiences to learn from per training step
REPLAY_SIZE = 10_000  # buffer size of transitions to learn from (should be large)
LEARNING_RATE = 10 ** -4  # learning rate of Adam optimizer
SYNC_TARGET_FRAMES = 1_000  # how often to update the target network
REPLAY_START_SIZE = 10_000  # wait for this many experiences before starting to learn
ALSO_CSV = True  # whether to save csv table of progress besides tensorboard

# epsilon-greedy parameters
EPSILON_DECAY_FRAMES = 150_000  # decay epsilon-greedy over this many frames
EPSILON_START = 1.0  # start with 100% chance of random action
EPSILON_END = 0.01  # end with almost no chance of random action


def main() -> None:
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--logdir", help="Path to logdir for tensorboard")
    parser.add_argument("--savedir", help="Where to save model weights")
    args = parser.parse_args()

    print("---Setting up---")
    # set up device
    if args.dev == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available. GPU is required but not accessible."
        print("CUDA is available.")
    device = torch.device(args.dev)
    print(f"Running on {device}")
    base_log_dir = Path(args.logdir)
    save_dir = Path(args.savedir)
    assert base_log_dir.is_dir(), f"Log directory {base_log_dir} does not exist"
    assert save_dir.is_dir(), f"Save directory {save_dir} does not exist"

    # make environment
    env = make_env(ENV_NAME)
    print(f"Observation space: {env.observation_space.shape}")

    # make q models (CNNs that predict state-action values)
    net = DQN(env.observation_space.shape, env.action_space.n)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n)
    net.to(device)
    tgt_net.to(device)
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
    log_dir = base_log_dir / f"{time.time():.0f}"
    print(f"Tensorboard output: {log_dir}")
    writer = SummaryWriter(log_dir=str(log_dir))
    last_loss = np.nan
    has_started = False
    if ALSO_CSV:
        csv_file = log_dir / "log.csv"
        first_write = True

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
            if not has_started:
                print("Required number of experiences reached - DQN training starts now.")
                has_started = True
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
                print(f"New best reward: {m_reward:.3f}")
                save_path = save_dir / f"best_{m_reward:.0f}.dat"
                torch.save(net.state_dict(), str(save_path))
                best_m_reward = m_reward

            # end training if our performance is good
            if m_reward > REWARD_LIMIT:
                print(f"Solved after {frame_idx + 1} frames!")
                break

            # log progress
            current_time = time.time()
            elapsed = current_time - last_time
            print(f"Frame {frame_idx:,}: {len(total_rewards)} games, {reward:.0f} last reward, {m_reward:.3f} running mean, {elapsed:.3f}s game time")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("games", len(total_rewards), frame_idx)
            writer.add_scalar("buffer_length", len(buffer), frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("elapsed", elapsed, frame_idx)
            writer.add_scalar("running_mean_reward", m_reward, frame_idx)
            writer.add_scalar("last_loss", last_loss, frame_idx)

            if ALSO_CSV:
                entry = {
                    "i_frame": frame_idx,
                    "epsilon": epsilon,
                    "games": len(total_rewards),
                    "buffer_length": len(buffer),
                    "last_loss": last_loss,
                    "elapsed": elapsed,
                    "running_mean_reward": m_reward,
                    "reward": reward,
                }
                entry = pd.DataFrame([entry])
                if first_write:
                    entry.to_csv(csv_file)
                    first_write = False
                else:
                    entry.to_csv(csv_file, header=False, mode="a")

            last_time = current_time

        # give tgt network latest weights
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        # update frame count
        frame_idx += 1


if __name__ == "__main__":
    main()
