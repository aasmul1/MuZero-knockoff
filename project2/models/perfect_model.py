import numpy as np
from models.super_model import Model
from typing import Dict, Tuple, List, Any

class PerfectModel(Model):
    """
    En modell som bruker de faktiske spillreglene for å utføre perfekt simulering.
    Brukes for tradisjonell MCTS.
    """
    
    def __init__(self, game_state_manager):
        """
        Initialiserer med en game state manager som kjenner reglene.
        
        Args:
            game_state_manager: Et GameStateManager-objekt som implementerer
                                metoder som get_legal_actions, get_next_state_and_reward,
                                is_terminal_state, og evaluate_state
        """
        self.gsm = game_state_manager
        
    def represent_state(self, observation):
        """
        For en perfekt modell er den abstrakte tilstanden den samme som den faktiske spilltilstanden.
        
        Args:
            observation: Aktuell observasjon fra miljøet (grid-representasjon)
            
        Returns:
            Samme observasjon, siden vi har perfekt informasjon
        """
        return observation
    
    def transition(self, abstract_state, action):
        """
        Bruker spillsimulatoren for å få neste tilstand og belønning.
        
        Args:
            abstract_state: Nåværende spilltilstand
            action: Handlingen som skal utføres (int)
            
        Returns:
            Tuple av (next_state, reward)
        """
        next_state, reward, _ = self.gsm.get_next_state_and_reward(abstract_state, action)
        return next_state, reward
    
    def predict(self, abstract_state):
        """
        For en perfekt modell i MCTS bruker vi heuristikker for å evaluere tilstanden.
        
        Args:
            abstract_state: Spilltilstanden som skal evalueres
            
        Returns:
            Tuple av (policy, value) hvor:
            - policy er en dict som mapper actions til sannsynligheter
            - value er en heuristisk vurdering av tilstanden
        """
        # Sjekk om dette er en terminal tilstand
        if self.gsm.is_terminal_state(abstract_state):
            # Tom policy for terminal tilstander
            return {}, self.gsm.evaluate_state(abstract_state)
        
        # Hent lovlige handlinger
        legal_actions = self.gsm.get_legal_actions(abstract_state)
        
        # Lag uniform policy over lovlige handlinger
        policy = {}
        probability = 1.0 / len(legal_actions) if legal_actions else 0
        for action in legal_actions:
            policy[action] = probability
        
        # Bruk heuristisk evaluering for value
        value = self.gsm.evaluate_state(abstract_state)
        
        return policy, value