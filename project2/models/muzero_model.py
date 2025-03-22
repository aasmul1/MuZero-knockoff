import numpy as np
from models.super_model import Model

class MuZeroModel(Model):
    """
    en lært modell som bruker nevrale nettverk til å forutsi tilstandsoverganger,
    belønninger, policies og verdier
    """
    
    def __init__(self, representation_network, dynamics_network, prediction_network, action_space_size):
        """
        initialiserer med de tre nevrale nettverkene som brukes i MuZero
        """
        self.representation_network = representation_network
        self.dynamics_network = dynamics_network
        self.prediction_network = prediction_network
        self.action_space_size = action_space_size
    
    def represent_state(self, observation):
        """
        bruker representation-nettverket til å kode observasjonen
        """
        processed_observation = self._preprocess_observation(observation)
        abstract_state = self.representation_network.predict(processed_observation)
        return abstract_state
    
    def transition(self, abstract_state, action):
        """
        bruker dynamics-nettverket til å forutsi neste abstrakte tilstand og belønning
        """
        encoded_action = self._encode_action(action)
        dynamics_input = self._combine_state_action(abstract_state, encoded_action)
        next_state, reward = self.dynamics_network.predict(dynamics_input)
        return next_state, reward
    
    def predict(self, abstract_state):
        """
        bruker prediction-nettverket til å forutsi policy og value
        """
        policy_logits, value = self.prediction_network.predict(abstract_state)
        policy = self._softmax(policy_logits)
        return policy, value
    
    # hjelpemetoder
    def _preprocess_observation(self, observation):
        return observation
    
    def _encode_action(self, action):
        encoded = np.zeros(self.action_space_size)
        encoded[action] = 1.0
        return encoded
    
    def _combine_state_action(self, state, action):
        return np.concatenate([state, action], axis=-1)
    
    def _softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)