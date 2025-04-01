import numpy as np

from project2.Action import Action


class Node:
    def __init__(self, hidden_state):
        self.hidden_state: np.ndarray = hidden_state
        self.parent_node: Node | None = None  # The parent node that leads to this node
        self.parent_action: Action | None = None  # The parent node's action taken to get to this node
        self.available_actions: list[Action] | None = None

    def __str__(self):
        # return f"Node's hidden state:\n {self.hidden_state}"
        return f"Node with hidden state size {self.hidden_state.size}"

    def __hash__(self):
        return hash(self.hidden_state.tobytes())

    def __eq__(self, other):
        if isinstance(other, Node):
            # Check equality based on hidden_state
            return np.array_equal(self.hidden_state, other.hidden_state)
        return False

    def set_available_actions(self, available_actions: list[Action]):
        self.available_actions = available_actions

    def get_available_actions(self) -> list[Action]:
        assert self.available_actions is not None, f"No available actions, state:\n{self.hidden_state}"
        return self.available_actions
