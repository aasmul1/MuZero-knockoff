

class BufferGame:


    def __init__(self):
        self.observatios = []
        self.actions = []
        self.rewards = []
        self.search_policies = []
        self.values = []

    def store_search_statistics(self, visit_counts, root_node):
        """
        store stats for this position
        """
        
        sum_visits = sum(visit_counts.values())
        
        policy = {action: visits/sum_visits for action, visits in visit_counts.items()}




    def append(self, observation, action = None, reward  = None, value = None):
        """
        Appends one step to the trajectory
        """

        self.observations.append(observation)

        if action is not None:
            self.actions.append(action)

        if reward is not None:
            self.rewards.append(reward)

        if value is not None:
            self.values.append(value)


