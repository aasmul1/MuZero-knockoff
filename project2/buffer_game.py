import torch

class Game:
    """
    Class to store a complete game trajectory for the replay buffer.
    """
    def __init__(self):
        self.observations = []  # Game observations
        self.actions = []  # Actions taken
        self.rewards = []  # Rewards received
        self.search_policies = []  # MCTS policy distributions
        self.values = []  # Value estimates

    def store_search_statistics(self, visit_counts, root_node):
        """
        Store the MCTS search statistics for this position.
        
        Args:
            visit_counts: Dictionary of visit counts for each action
            root_node: The root node of the search
        """
        sum_visits = sum(visit_counts.values())
        
        # Avoid division by zero
        if sum_visits == 0:
            # Create uniform policy if no visits
            policy = {action: 1.0 / len(visit_counts) for action in visit_counts.keys()}
        else:
            # Normalize visit counts to create a policy
            policy = {action: visits/sum_visits for action, visits in visit_counts.items()}
        
        self.search_policies.append(policy)

    def append(self, observation, action=None, reward=None, value=None):
        """
        Append one step to the trajectory.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            value: Value estimate
        """
        # Ensure observation is float32 if it's a tensor
        if isinstance(observation, torch.Tensor):
            if observation.dtype != torch.float32:
                observation = observation.to(dtype=torch.float32)
            # Create a clone to avoid reference issues
            observation = observation.clone()
            
        self.observations.append(observation)
        
        if action is not None:
            self.actions.append(action)
            
        if reward is not None:
            self.rewards.append(reward)
            
        if value is not None:
            self.values.append(value)


