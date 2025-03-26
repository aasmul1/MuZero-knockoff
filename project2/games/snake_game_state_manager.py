from snake import SnakeGame
from generic_game_state_manager import GameStateManager

import numpy as np


class SnakeGameStateManager(GameStateManager):
    """adapter-class for SnakeGame for å passe Game State Manager-grensesnittet"""
    
    def __init__(self, grid_size=10, grow_on_food=False):
        self.game = SnakeGame(grid_size, grow_on_food)
        self.state_cache = {}  # for caching av tilstandsoverganger
    
    def generate_initial_state(self):
        """ initial gamestate"""
        return self.game.reset()
    
    def get_legal_actions(self, state):
        """ alle lovlige actions per state"""
        temp_game = self._create_temp_game(state)
        return temp_game.get_legal_actions()
    
    def get_next_state_and_reward(self, state, action):
        """returner neste state og reward gitt state og action"""
        # sjekk cache først
        cache_key = (state.tobytes(), action)
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]
        
        # lag midlertidig spill med denne stateb
        temp_game = self._create_temp_game(state)
        
        # utfør action
        next_state, reward, done = temp_game.step(action)
        
        # cache resultatet
        self.state_cache[cache_key] = (next_state, reward, done)
        
        return next_state, reward, done
    
    def is_terminal_state(self, state):
        """sjekk om state er terminal"""
        temp_game = self._create_temp_game(state)
        return temp_game.done
    
    def evaluate_state(self, state):
        """ evaluering for en state"""
        temp_game = self._create_temp_game(state)
        if temp_game.done:
            return -10
        
        # evaluer basert på avstand til frukt
        head = temp_game.snake[0]
        fruit = temp_game.fruit
        distance = abs(head[0] - fruit[0]) + abs(head[1] - fruit[1])
        
        # normaliser avstand til [-1, 1] område
        max_distance = 2 * temp_game.grid_size
        normalized_distance = -2 * (distance / max_distance) + 1
        
        return normalized_distance
    
    def _create_temp_game(self, state):
        """ midlertidig spill med gitt state"""
        temp_game = SnakeGame(self.game.grid_size, self.game.grow_on_food)
        
        # hent slange- og fruktposisjoner fra tilstanden
        snake_cells = np.where(state == 1)
        fruit_pos = np.where(state == 2)
        
        if len(snake_cells[0]) > 0:
            temp_game.snake = [(snake_cells[0][i], snake_cells[1][i]) 
                              for i in range(len(snake_cells[0]))]
        
        if len(fruit_pos[0]) > 0:
            temp_game.fruit = (fruit_pos[0][0], fruit_pos[1][0])
        
        temp_game.grid = state.copy()
        
        return temp_game