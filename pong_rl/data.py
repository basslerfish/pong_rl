"""

"""
import collections
from dataclasses import dataclass

import numpy as np
import torch

# type declarations
State = np.ndarray
Action = int
BatchTensors = tuple[
    torch.ByteTensor,  # current state
    torch.LongTensor,  # actions
    torch.Tensor,      # rewards
    torch.BoolTensor,  # done / truncated
    torch.ByteTensor,  # next state
]


@dataclass
class Experience:
    """
    Keep s, a, r, s'.
    Could also be called a transition.
    """
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    """Keep experiences in a buffer.
    We use this for sampling for SGD later.
    """
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        sampled = [self.buffer[i] for i in indices]
        return sampled


def batch_to_tensors(batch: list[Experience], device: torch.device) -> BatchTensors:
    """
    Convert a batch of experiences into tensors for training.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    new_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done_trunc)
        new_states.append(exp.new_state)

    states = torch.as_tensor(np.asarray(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.BoolTensor(dones)
    new_states = torch.as_tensor(np.asarray(new_states))

    all_tensors = (states, actions, rewards, dones, new_states)
    for tensor in all_tensors:
        tensor.to(device)
    return all_tensors
