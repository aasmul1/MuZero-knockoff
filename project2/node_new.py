class NodeNew:
    def __init__(self, hidden_state):
        self.hidden_state = hidden_state
        self.parent_node = None  # The parent node that leads to this node
        self.parent_action = None  # The parent node's action taken to get to this node

    def __str__(self):
        print("Node's hidden state:", self.hidden_state)
