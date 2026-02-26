"""
Agent classs to choose actions in environment.
"""
import torch
import gymnasium as gym
import numpy as np

from pong_rl.data import ExperienceBuffer, Experience
from pong_rl.model import DQN


class Agent:
    """
    RL agent that will use DQN to select actions.
    """
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer) -> None:
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: np.ndarray | None = None
        self.total_reward: float | None = None

        self._reset()

    def _reset(self) -> None:
        """Reset state and reward."""
        self.state, _ = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(
            self,
            net: DQN,
            device: torch.device,
            epsilon: float = 0.0,
    ) -> float | None:
        """
        Take a single step in the environment.
        Implements epsilon-greedy, so either random action or DQN based action.
        """
        done_reward = None

        # choose action
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.choose_action(net, device)

        # take step
        new_state, reward, is_done, is_trunc, _ = self.env.step(action)
        self.total_reward += reward

        # save transition
        exp = Experience(
            state=self.state,
            action=action,
            reward=float(reward),
            done_trunc=is_done or is_trunc,
            new_state=new_state,
        )
        self.exp_buffer.append(exp)

        # update
        self.state = new_state
        if is_done or is_trunc:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def choose_action(self, net: DQN, device: torch.device) -> int:
        """
        Feedforward pass through our deep q network.
        Followed by selection of action with highest state-action value.
        """
        state_tensor = torch.as_tensor(self.state)
        state_tensor = state_tensor.to(device)
        state_tensor = state_tensor.unsqueeze(0)  # add batch dim
        q_vals = net(state_tensor)
        _, action = torch.max(q_vals, dim=1)  # return best_q and i_best_q
        action = int(action.item())
        return action
