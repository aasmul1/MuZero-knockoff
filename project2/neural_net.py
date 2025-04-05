from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkOutput:
    def __init__(self, value, reward, policy_logits, hidden_state):
        self.value = value
        self.reward = reward
        self.policy_logits = policy_logits
        self.hidden_state = hidden_state


class MuZeroNetwork(nn.Module):
    def __init__(self, observation_dim, action_space, hidden_size, latent_dim):
        """
        Args:
            observation_dim (int): Dimension of the flattened observation.
            action_space (iterable): Iterable representing the action space.
            hidden_size (int): Number of neurons in hidden layers.
            latent_dim (int): Size of the latent (abstract) state.
        """
        super(MuZeroNetwork, self).__init__()
        # Handle both cases: action_space can be an integer or an iterable
        if isinstance(action_space, (int, float)):
            self.action_space_size = action_space
        else:
            self.action_space_size = len(action_space)

        self.latent_dim = latent_dim
        self._training_steps = 0

        # Representation network: converts raw observation to latent state.
        self.representation_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
        )

        # Prediction network: from latent state to (policy_logits, value).
        # The network outputs a vector of size (action_space_size + 1), where the first
        # action_space_size elements are policy logits and the final element is the value.
        self.prediction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_space_size + 1)
        )

        # Dynamics network: from (latent state, action) to (next latent state, reward).
        # The input is the concatenation of the latent state and a one-hot encoded action.
        # The output is a vector of size (latent_dim + 1): the first latent_dim values are
        # the new latent state and the final value is the predicted reward.
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + self.action_space_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim + 1)
        )

    def represent_state(self, observation):
        """
        Converts a raw observation into a latent state.
        """
        return self.representation_net(observation)

    def predict(self, abstract_state):
        """
        Given a latent state, outputs the policy logits and value.
        """
        output = self.prediction_net(abstract_state)
        policy_logits = output[:, :self.action_space_size]
        value = output[:, self.action_space_size]
        return policy_logits, value

    def transition(self, abstract_state, action):
        """
        Given a latent state and an action, predicts the next latent state and reward.
        """
        action_one_hot = F.one_hot(action, num_classes=self.action_space_size).float()
        x = torch.cat([abstract_state, action_one_hot], dim=-1)
        dynamics_out = self.dynamics_net(x)
        next_state = dynamics_out[:, :self.latent_dim]
        reward = dynamics_out[:, self.latent_dim]
        return next_state, reward

    def initial_inference(self, observation) -> NetworkOutput:
        """
        Runs representation and prediction: from raw observation to latent state, then
        predicts the policy and value. Reward is set to zero at the initial inference.
        """
        abstract_state = self.represent_state(observation)
        policy_logits, value = self.predict(abstract_state)
        reward = torch.zeros_like(value)
        return NetworkOutput(value, reward, policy_logits, abstract_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        """
        Given a latent state and an action, applies the dynamics network to get the next
        latent state and reward, then uses the prediction network on that new latent state.
        """
        next_state, reward = self.transition(hidden_state, action)
        policy_logits, value = self.predict(next_state)
        return NetworkOutput(value, reward, policy_logits, next_state)

    def get_weights(self) -> Dict:
        """
        Returns the current network weights.
        """
        return self.state_dict()

    def training_steps(self) -> int:
        """
        Returns the number of training steps (batches) performed.
        """
        return self._training_steps

    def increment_training_steps(self):
        """
        Increments the training step counter. This should be called after each training update.
        """
        self._training_steps += 1
