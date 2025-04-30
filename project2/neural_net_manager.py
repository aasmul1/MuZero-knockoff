import logging
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torch.optim as optim

from project2.neural_net import MuZeroNetwork, NetworkOutput


def _get_device():
    if os.name == 'nt':
        import torch_directml
        device = torch_directml.device(torch_directml.default_device())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


class NeuralNetManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.device = _get_device()
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)
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
        observation = observation.to(self.device, dtype=torch.float32)
        return self.model.initial_inference(observation)

    def recurrent_inference(self, abstract_state: torch.Tensor, action: torch.Tensor) -> NetworkOutput:
        hidden_state = abstract_state.to(self.device, dtype=torch.float32)
        action = action.to(self.device)
        return self.model.recurrent_inference(hidden_state, action)

    def get_weights(self) -> Dict:
        return self.model.get_weights()

    def get_training_steps(self) -> int:
        return self.model.training_steps()

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        policy_loss, value_loss, reward_loss = self.compute_loss(batch)
        total_loss = policy_loss + value_loss + reward_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.model.increment_training_steps()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item()
        }

    def compute_loss(self, batch):
        if any(len(batch[key]) == 0 for key in batch):
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        observations = batch["observations"].to(self.device, dtype=torch.float32)
        target_policies = batch["target_policy"].to(self.device, dtype=torch.float32)
        target_rewards = batch["target_reward"].to(self.device, dtype=torch.float32)
        target_values = batch["target_value"].to(self.device, dtype=torch.float32)
        actions = batch["actions"].to(self.device)

        out = self.model.initial_inference(observations)

        log_softmax_policies = F.log_softmax(out.policy_logits, dim=1)

        policy_loss = 0.0
        if target_policies.size(0) > 0 and target_policies.size(1) > 0:
            first_policy = target_policies[:, 0, :]
            policy_loss = -(first_policy * log_softmax_policies).sum(dim=1).mean()

        value_loss = 0.0
        if target_values.size(0) > 0 and target_values.size(1) > 0:
            value_loss = F.mse_loss(out.value, target_values[:, 0])

        reward_loss = 0.0

        hidden_state = out.hidden_state

        if actions.size(0) > 0 and actions.size(1) > 0:
            max_unroll = min(actions.size(1),
                             target_values.size(1) - 1 if target_values.size(1) > 0 else 0,
                             target_rewards.size(1) if target_rewards.size(1) > 0 else 0)

            if max_unroll > 0:
                for k in range(max_unroll):
                    action_indices = actions[:, k]

                    if k >= target_values.size(1) - 1 or k >= target_rewards.size(1):
                        break

                    out = self.model.recurrent_inference(hidden_state, action_indices)

                    value_loss += F.mse_loss(out.value, target_values[:, k + 1])

                    reward_loss += F.mse_loss(out.reward, target_rewards[:, k])

                    hidden_state = out.hidden_state

        return policy_loss, value_loss, reward_loss
