"""
Deep Q network and loss function.
"""
import typing as tt

import torch
from torch import nn

from pong_rl.data import Experience, batch_to_tensors


class DQN(nn.Module):
    """
    Simple convnet to process atari game data.
    Will be trained to approximate Q with value updating.
    """
    def __init__(self, input_shape: tt.Any, n_actions: int) -> None:
        super().__init__()
        n_channels = input_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # mock pass through
        # to conv output size
        mock_data = torch.zeros(1, *input_shape)
        mock_out = self.conv(mock_data)
        out_size = mock_out.size()[-1]

        self.fc = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor) -> torch.Tensor:
        """
        Takes state tensor, outputs q-values of all actions.
        We scale before feedforward pass.
        We could have done this in a wrapper, but it's more efficient to keep data as uint8
        as long as possible.
        """
        xx = x / 255.0
        out = self.conv(xx)
        out = self.fc(out)
        return out


def calculate_loss(
        batch:list[Experience],
        net: DQN,
        tqt_net: DQN,
        device: torch.device,
        gamma: float,
) -> torch.Tensor:
    """
    Calculate Q loss for a list of experiences.
    The loss is the MSE between the predicted state-action value
    and the Bellman update (update = r + gamma * Q(s', a')).

    Uses a tgt_net for Q(s', a').
    """
    tensors = batch_to_tensors(batch, device=device)

    # let's predict the state action values
    states = tensors[0]
    actions = tensors[1]
    rewards = tensors[2]
    dones = tensors[3]
    next_states = tensors[4]

    # compute q-values of states
    q_values = net(states)  # all actions
    q_values_chosen = q_values.gather(1, actions.unsqueeze(-1))  # only of actions we chose

    # compute q-values of next states
    with torch.no_grad():
        q_values_next = tqt_net(next_states)
        q_values_next_optimal = q_values_next.max(1)[0]  # q vals of best action in next state
        q_values_next_optimal[dones] = 0.0  # no value if next state is end
        q_values_next_optimal = q_values_next_optimal.detach()

    # compute bellman update of q values
    expected_q_values = rewards + gamma * q_values_next_optimal

    # use bellman update for loss to nudge weights
    loss = nn.MSELoss()(q_values_chosen, expected_q_values)
    return loss
