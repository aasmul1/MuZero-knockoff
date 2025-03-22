import numpy as np
from models.super_model import Model

class PerfectModel(Model):
    """
    en modell som bruker de faktiske spillreglene for å utføre perfekt simulering
    brukes for tradisjonell MCTS
    """
    
    def __init__(self, game_simulator):
        """
        initialiserer med en spillsimulator som kjenner reglene
        
        args:
            game_simulator: et objekt som kan simulere spillet
                           (trenger metoder: get_legal_actions, step)
        """
        self.simulator = game_simulator
    
    def represent_state(self, observation):
        """
        for en perfekt modell er den abstrakte tilstanden den samme som den faktiske spilltilstanden
        """
        return observation
    
    def transition(self, abstract_state, action):
        """
        bruker spillsimulatoren for å få neste tilstand og belønning
        """
        # kloner tilstanden for å unngå å endre originalen
        next_state = self.simulator.clone_state(abstract_state)
        reward = self.simulator.step(next_state, action)
        return next_state, reward
    
    def predict(self, abstract_state):
        """
        for en perfekt modell i MCTS kan vi bruke enkle heuristikker
        """
        legal_actions = self.simulator.get_legal_actions(abstract_state)
        # uniform policy over lovlige handlinger
        policy = np.zeros(self.simulator.action_space_size)
        policy[legal_actions] = 1.0 / len(legal_actions)
        
        # enkel heuristisk verdifunksjon
        value = self.simulator.get_heuristic_value(abstract_state)
        
        return policy, value