import logging
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim

from project2.neural_net import MuZeroNetwork, NetworkOutput


def _get_device():
    if os.name == 'nt':  # Windows
        import torch_directml
        # Will use AMD GPU if available and supported, else cpu
        device = torch_directml.device(torch_directml.default_device())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


class NeuralNetManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.device = _get_device()
        # Set default tensor type to float32
        torch.set_default_tensor_type(torch.FloatTensor)
        self.model = MuZeroNetwork(
            observation_dim=self.config.OBSERVATION_DIM,
            action_space=self.config.ACTION_SPACE,
            hidden_size=self.config.HIDDEN_SIZE,
            latent_dim=self.config.LATENT_DIM
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        self._training_steps = 0
        self.logger.info(f"Using device {self.device}")

    def initial_inference(self, observation: torch.Tensor) -> NetworkOutput:
        # Ensure observation is float32
        observation = observation.to(self.device, dtype=torch.float32)
        return self.model.initial_inference(observation)

    def recurrent_inference(self, abstract_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        # Ensure abstract_state is float32
        hidden_state = abstract_state.to(self.device, dtype=torch.float32)
        action = action.to(self.device)
        return self.model.recurrent_inference(hidden_state, action)

    def get_weights(self) -> Dict:
        """
        Returns the current network weights.
        """
        return self.model.get_weights()

    def get_training_steps(self) -> int:
        """
        Return the current trainig steps
        """
        return self.model.training_steps()

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.model.increment_training_steps()
        return loss.item()

    def compute_loss(self, batch):
        # Ensure all tensors are float32
        observations = batch["observations"].to(self.device, dtype=torch.float32)
        target_policies = batch["target_policy"].to(self.device, dtype=torch.float32)
        target_rewards = batch["target_reward"].to(self.device, dtype=torch.float32)
        target_values = batch["target_value"].to(self.device, dtype=torch.float32)
        actions = batch["actions"].to(self.device)  # Keep as long for indices

        out = self.model.initial_inference(observations)

        # Use KL divergence loss for policy - target_policies are now probability distributions
        log_softmax_policies = F.log_softmax(out.policy_logits, dim=1)
        loss_policy = -(target_policies * log_softmax_policies).sum(dim=1).mean()

        loss_value = F.mse_loss(out.value, target_values[:, 0])
        total_reward_loss = 0.0

        hidden_state = out.hidden_state

        max_unroll = min(actions.size(1), target_values.size(1) - 1, target_rewards.size(1))
        for k in range(max_unroll):

            out = self.model.recurrent_inference(hidden_state, actions[:, k])

            # Only value and reward losses during recurrent steps
            loss_value += F.mse_loss(out.value, target_values[:, k + 1])
            min_size = min(out.reward.shape[0], target_rewards[:, k].shape[0])
            total_reward_loss += F.mse_loss(out.reward[:min_size], target_rewards[:min_size, k])
            hidden_state = out.hidden_state


        total_loss = loss_policy + loss_value + total_reward_loss
        return total_loss
