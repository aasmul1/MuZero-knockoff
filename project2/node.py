import numpy as np
import torch

from project2.Action import Action


class Node:
    def __init__(self, hidden_state):
        # Convert tensor to float32 if it's a tensor
        if isinstance(hidden_state, torch.Tensor):
            self.hidden_state = hidden_state.to(dtype=torch.float32)
        else:
            self.hidden_state = hidden_state
        self.parent_node: Node | None = None  # The parent node that leads to this node
        self.parent_action: Action | None = None  # The parent node's action taken to get to this node
        self.available_actions: list[Action] | None = None

    def __str__(self):
        # return f"Node's hidden state:\n {self.hidden_state}"
        # return f"Hidden state size {self.hidden_state.size}"
        return f"{self.hidden_state}"

    def __hash__(self):
        if isinstance(self.hidden_state, np.ndarray):
            return hash(self.hidden_state.tobytes())
        elif isinstance(self.hidden_state, torch.Tensor):
            try:
                # Ensure tensor is detached, on CPU, and in float32 before conversion
                tensor = self.hidden_state.detach()
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                # Convert to float32 if it's not already
                if tensor.dtype != torch.float32:
                    tensor = tensor.to(dtype=torch.float32)
                
                # Convert to numpy and hash the bytes
                return hash(tensor.numpy().tobytes())
            except Exception as e:
                # Fallback to string representation if conversion fails
                return hash(str(self.hidden_state))
        else:
            return hash(str(self.hidden_state))

    def __eq__(self, other):
        if isinstance(other, Node):
            return hash(self) == hash(other)
        return False

    def set_available_actions(self, available_actions: list[Action]):
        self.available_actions = available_actions

    def get_available_actions(self) -> list[Action]:
        assert self.available_actions is not None, "No available actions for this node"
        return self.available_actions
