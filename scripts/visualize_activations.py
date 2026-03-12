"""
Visualize activations of the DQN network while it plays Pong.
Recommended e.g. by Jeremy Howard from FastAi to understand what the network does.

We rely on .register_forward_hook()
"""
from collections import deque
from functools import partial
from pathlib import Path

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax

from pong_rl.wrappers import make_env
from pong_rl.model import DQN
from pong_rl.agent import Agent
from pong_rl.data import ExperienceBuffer

# params
ENV_NAME = "PongNoFrameskip-v4"
WEIGHTS_FILE = Path("/Users/mathis/Code/github/pong_rl/data/best_weights.dat")

# make env
env = make_env(ENV_NAME, render_mode="rgb_array")
action_names = env.unwrapped.get_action_meanings()
print(action_names)

# instantiate model
device = torch.device("cpu")  # small enough to run on cpu
net = DQN(env.observation_space.shape, env.action_space.n)
print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")

# load weights
weights_file = WEIGHTS_FILE
assert weights_file.is_file(), f"Weights file {weights_file} does not exist."
state_dict = torch.load(str(weights_file), map_location=device)
net.load_state_dict(state_dict)

# add a hook to each layer of the module
activations = {}

def hook(this_module, this_input, this_output, module_name: str) -> None:
    """Place activation in a dict"""
    activations[module_name] = this_output.detach().cpu().numpy()

for name, module in net.named_modules():
    fn = partial(hook, module_name=name)
    module.register_forward_hook(fn)

# prep agent
buffer = ExperienceBuffer(0)  # we don't need to save experiences now
agent = Agent(env, buffer)

# play episode and render
episode_reward = None

# prep plot
plt.ion()
fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12, 4))

count = 0
act_hist = deque(maxlen=6)
action_hist = deque(maxlen=6)

# iterate over frames, make interactive plot
while episode_reward is None:
    episode_reward = agent.play_step(net, device=device, epsilon=0)
    frame = env.render()
    if count == 0:
        for key, val in activations.items():
            print(key, val.shape)

    for ax in axes:
        ax.clear()

    ax = axes[0]
    ax.set_title("Frame")
    ax.imshow(frame)

    ax = axes[1]
    data = activations["fc.1"]
    data = np.squeeze(data)
    ax.plot(data)
    ax.set_ylim(-0.1, 2)

    # plot history
    act_hist.append(data)
    m_data = list(act_hist)
    m_data = np.stack(m_data, axis=0)
    m = np.mean(m_data, axis=0)
    ax.plot(m, alpha=0.5)

    # output layer
    ax = axes[2]
    data = activations["fc.2"]
    data = np.squeeze(data)
    ax.plot(data)
    ax.scatter(np.argmax(data), np.max(data), color="red")

    # plot history
    action_hist.append(data)
    m_data = list(action_hist)
    m_data = np.stack(m_data, axis=0)
    m = np.mean(m_data, axis=0)
    ax.plot(m, alpha=0.5)

    # formatting
    ax.set_title("Output layer activations")
    ax.set_xlabel("Action")
    ax.set_ylabel("Action probs")
    ax.set_ylim(-1, 3)
    ax.set_xticks(np.arange(len(action_names)), action_names, rotation=45)
    ax.set_xlim(-1, 6)

    # draw
    plt.draw()
    plt.pause(0.1)

    count += 1
plt.show()
