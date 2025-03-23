class MCTSNew:
    def __init__(self, c_1: float = 1.25, c_2: float = 19.652, simulations: int = 800, discount: float = 0.997):
        # Used to control the influence of policy relative to the value as nodes are visited more often
        self.c_1 = c_1
        self.c_2 = c_2

        self.simulations = simulations  # Number of simulations to run per search
        self.discount = discount

        pass

    def _select_action(self, current_state, edges):
        pass

    def _expand_node(self):
        pass
