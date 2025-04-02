class Edge:
    def __init__(self, policy, mean_value_q: float = 0, visit_count: int = 0):
        self.policy = policy
        self.mean_value_q = mean_value_q
        self.visit_count = visit_count

        self.reward = None
        self.state_transition = None

    def set_reward(self, reward):
        self.reward = reward

    def set_state_transition(self, state_transition):
        self.state_transition = state_transition  # The state the edge leads to
