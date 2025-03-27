from catch import CatchGame
from generic_game_state_manager import GameStateManager

import numpy as np

class CatchGameStateManager(GameStateManager):
    """Adapter klasse for CatchGame som passer game state manager interface"""
    
    def __init__(self, grid_width=10, grid_height=10):
        self.game = CatchGame(grid_width, grid_height)
        self.state_cache = {}  # for caching av state transitions
    
    def generate_initial_state(self):
        """generer initial game state"""
        return self.game.reset()
    
    def get_legal_actions(self, state):
        """returnerer alle lovlige actions for en state"""
        # lag temp game med denne staten for action kalkulering
        temp_game = self._create_temp_game(state)
        return temp_game.get_legal_actions()
    
    def get_next_state_and_reward(self, state, action):
        """returnerer neste state og reward gitt state og action"""
        # sjekk cache først
        cache_key = (state.tobytes(), action) 
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]
        
        # lag temp game med denne staten
        temp_game = self._create_temp_game(state)
        
        # utfør action
        next_state, reward, done = temp_game.step(action)
        
        # cache resultatet
        self.state_cache[cache_key] = (next_state, reward, done)
        
        return next_state, reward, done
    
    
    def is_terminal_state(self, state):
        """sjekk om state er terminal"""
        # lag temp game med denne staten for å sjekke om terminal
        temp_game = self._create_temp_game(state)
        return temp_game.game_over
    
    def evaluate_state(self, state):
        """enkel heuristisk evaluering for en state"""
        # for catch, en enkel heuristikk kan være:
        # - avstanden mellom frukt og paddle (nærmere er bedre)
        temp_game = self._create_temp_game(state)
        if temp_game.game_over:
            return -10
        
        fruit_col = temp_game.fruit_pos[1]
        paddle_center = temp_game.paddle_pos + temp_game.paddle_size // 2
        distance = abs(fruit_col - paddle_center)
        
        # normaliser avstand til [-1, 1] range
        max_distance = temp_game.width
        normalized_distance = -2 * (distance / max_distance) + 1
        
        return normalized_distance
    
    def _create_temp_game(self, state):
        """lag et midlertidig game med gitt state"""
        temp_game = CatchGame(self.game.width, self.game.height, self.game.paddle_size)
        
        # finn paddle og fruit posisjoner fra staten
        paddle_indices = np.where(state[-1] == 1)[0]
        if len(paddle_indices) > 0:
            temp_game.paddle_pos = paddle_indices[0]
        
        fruit_pos = np.where(state == 2)
        if len(fruit_pos[0]) > 0:
            temp_game.fruit_pos = [fruit_pos[0][0], fruit_pos[1][0]]
        
        temp_game.grid = state.copy()
        
        return temp_game