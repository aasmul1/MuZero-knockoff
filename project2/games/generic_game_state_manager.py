class GameStateManager:
    """abstrakt interface for spilltilstandsforvaltere"""
    
    def generate_initial_state(self):
        """generer initial game state"""
        raise NotImplementedError
    
    def get_legal_actions(self, state):
        """returner alle lovlige actions for en state"""
        raise NotImplementedError
    
    def get_next_state_and_reward(self, state, action):
        """returner neste state og reward gitt state og action"""
        raise NotImplementedError
    
    def is_terminal_state(self, state):
        """sjekk om state er terminal"""
        raise NotImplementedError
    
    def evaluate_state(self, state):
        """heuristisk evaluering av en state"""
        raise NotImplementedError