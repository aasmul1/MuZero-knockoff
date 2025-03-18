from catch import CatchGame

import numpy as np

class CatchGameStateManager:
    """Adapter class for CatchGame to fit the Game State Manager interface."""
    
    def __init__(self, grid_width=10, grid_height=10):
        self.game = CatchGame(grid_width, grid_height)
        self.state_cache = {}  # For caching state transitions
    
    def generate_initial_state(self):
        """Generate initial game state."""
        return self.game.reset()
    
    def get_legal_actions(self, state):
        """Return all legal actions for a state."""
        # Create temp game with this state for action calculation
        temp_game = self._create_temp_game(state)
        return temp_game.get_legal_actions()
    
    def get_next_state_and_reward(self, state, action):
        """Return next state and reward given state and action."""
        # Check cache first
        cache_key = (str(state.tobytes()), action)
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]
        
        # Create temp game with this state
        temp_game = self._create_temp_game(state)
        
        # Execute action
        next_state, reward, done = temp_game.step(action)
        
        # Cache result
        self.state_cache[cache_key] = (next_state, reward, done)
        
        return next_state, reward, done
    
    def is_terminal_state(self, state):
        """Check if state is terminal."""
        # Create temp game with this state to check if terminal
        temp_game = self._create_temp_game(state)
        return temp_game.game_over
    
    def evaluate_state(self, state):
        """Simple heuristic evaluation for a state."""
        # For catch, a simple heuristic could be:
        # - Distance between fruit and paddle (closer is better)
        temp_game = self._create_temp_game(state)
        if temp_game.game_over:
            return -10
        
        fruit_col = temp_game.fruit_pos[1]
        paddle_center = temp_game.paddle_pos + temp_game.paddle_size // 2
        distance = abs(fruit_col - paddle_center)
        
        # Normalize distance to [-1, 1] range
        max_distance = temp_game.width
        normalized_distance = -2 * (distance / max_distance) + 1
        
        return normalized_distance
    
    def _create_temp_game(self, state):
        """Create a temporary game with the given state."""
        temp_game = CatchGame(self.game.width, self.game.height, self.game.paddle_size)
        
        # Find paddle and fruit positions from the state
        paddle_indices = np.where(state[-1] == 1)[0]
        if len(paddle_indices) > 0:
            temp_game.paddle_pos = paddle_indices[0]
        
        fruit_pos = np.where(state == 2)
        if len(fruit_pos[0]) > 0:
            temp_game.fruit_pos = [fruit_pos[0][0], fruit_pos[1][0]]
        
        temp_game.grid = state.copy()
        
        return temp_game