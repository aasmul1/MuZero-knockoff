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


def safe_one_hot(tensor, num_classes):
    """
    Create a one-hot tensor that works with DirectML by avoiding scatter operations.
    
    Args:
        tensor: Input tensor with class indices
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        One-hot encoded tensor
    """
    # Convert tensor to int64 to ensure it contains valid indices
    if tensor.dtype != torch.int64 and tensor.dtype != torch.long:
        tensor = tensor.to(torch.int64)

    # Check if we're on CPU - if so, use the standard function
    if tensor.device.type == "cpu":
        return F.one_hot(tensor, num_classes=num_classes).float()  # Ensure float32

    # For DirectML or other devices, manually create one-hot tensor
    # Get tensor shape and add one dimension for one-hot
    shape = list(tensor.shape)
    shape.append(num_classes)

    # Create a zero tensor with the right shape (explicitly float32)
    result = torch.zeros(shape, dtype=torch.float32, device=tensor.device)

    # Handle different tensor dimensions
    if len(tensor.shape) == 1:
        # For 1D tensors (batch of indices)
        for i in range(tensor.shape[0]):
            idx = tensor[i].item()
            result[i, idx] = 1.0
    else:
        # For 0D tensors (single index)
        idx = tensor.item()
        result[idx] = 1.0

    return result


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
        # Ensure float32
        if isinstance(observation, torch.Tensor) and observation.dtype != torch.float32:
            observation = observation.to(dtype=torch.float32)
        return self.representation_net(observation)

    def predict(self, hidden_state: torch.Tensor):
        out = self.prediction_net(hidden_state)
        policy_logits = out[:, :-1]
        value = out[:, -1]
        return policy_logits, value

    def transition(self, abstract_state, action):
        """
        Given a latent state and an action, predicts the next latent state and reward.
        """
        # Ensure abstract_state is float32
        if isinstance(abstract_state, torch.Tensor) and abstract_state.dtype != torch.float32:
            abstract_state = abstract_state.to(dtype=torch.float32)

        # Use the safe one-hot function to avoid DirectML scatter issues
        action_one_hot = safe_one_hot(action, num_classes=self.action_space_size)

        # Make sure the dimensions match for concatenation
        if len(action_one_hot.shape) > len(abstract_state.shape):
            # If action_one_hot has more dimensions, squeeze it
            action_one_hot = action_one_hot.squeeze(0)
        elif len(action_one_hot.shape) < len(abstract_state.shape):
            # If abstract_state has more dimensions, unsqueeze action_one_hot
            action_one_hot = action_one_hot.unsqueeze(0)

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
        # Ensure float32
        if isinstance(observation, torch.Tensor) and observation.dtype != torch.float32:
            observation = observation.to(dtype=torch.float32)

        abstract_state = self.represent_state(observation)
        policy_logits, value = self.predict(abstract_state)
        reward = torch.zeros_like(value)
        return NetworkOutput(value, reward, policy_logits, abstract_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        """
        Given a latent state and an action, applies the dynamics network to get the next
        latent state and reward, then uses the prediction network on that new latent state.
        """
        # Ensure float32
        if isinstance(hidden_state, torch.Tensor) and hidden_state.dtype != torch.float32:
            hidden_state = hidden_state.to(dtype=torch.float32)

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
