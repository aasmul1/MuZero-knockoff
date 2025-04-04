import numpy as np
from models.super_model import Model

from project2.games.catch_game_state_manager import CatchGameStateManager


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

    def predict(self, state):
        """
               For en perfekt modell i MCTS bruker vi heuristikker for å evaluere tilstanden.

               Args:
                   state:

               Returns:
                   Tuple av (policy, value) hvor:
                   - policy er en dict som mapper actions til sannsynligheter
                   - value er en heuristisk vurdering av tilstanden
        """

        #  TODO Implement predict function for other game state managers
        if not isinstance(self.gsm, CatchGameStateManager):
            raise NotImplementedError("Not implemented predict function of Perfect Model for this Game State Manager")

        fruit_pos = np.where(state == 2)
        paddle_row = state[-1]
        paddle_indices = np.where(paddle_row == 1)[0]

        actions = self.gsm.get_legal_actions(state)
        policy = {action: 0.0 for action in actions}

        if fruit_pos[1].size == 0 or paddle_indices.size == 0:
            # fallback: assign uniform if something is off
            for action in actions:
                policy[action] = 1 / len(actions)
        else:
            fruit_col = fruit_pos[1][0]
            paddle_left = paddle_indices[0]
            paddle_right = paddle_indices[-1]

            if fruit_col < paddle_left and 0 in actions:
                policy[0] = 1.0  # move left
            elif fruit_col > paddle_right and 2 in actions:
                policy[2] = 1.0  # move right
            elif 1 in actions:
                policy[1] = 1.0  # stay
            else:
                for action in actions:
                    policy[action] = 1 / len(actions)  # fallback uniform

        # print("Fruit:", fruit_col, "Paddle:", paddle_left, "-", paddle_right, "Policy:", policy)

        # For value: 0 unless it's terminal, or you want to use a heuristic
        value = self.gsm.evaluate_state(state)
        return policy, value
