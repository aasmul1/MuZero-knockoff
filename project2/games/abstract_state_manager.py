class AbstractStateManager:
    """manager for MuZero's abstrakte tilstander"""
    
    def __init__(self, representation_network, dynamics_network, prediction_network):
        self.representation_network = representation_network
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network
    
    def map_observation_to_abstract_state(self, observation_sequence):
        """bruk representation-nettverket til å mappe observasjonssekvensen til en abstrakt tilstand"""
        return self.representation_network.forward(observation_sequence)
    
    def get_policy_and_value(self, abstract_state):
        """bruk prediction-nettverket til å få policy og value for abstrakt tilstand"""
        return self.prediction_network.forward(abstract_state)
    
    def get_next_abstract_state_and_reward(self, abstract_state, action):
        """bruk dynamics-nettverket til å få neste abstrakte tilstand og reward"""
        return self.dynamics_network.forward(abstract_state, action)