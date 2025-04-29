import numpy as np
import torch  # Add torch import at the top

from project2.games.catch import CatchGame
from project2.games.generic_game_state_manager import GameStateManager


class CatchGameStateManager(GameStateManager):
    """Adapter klasse for CatchGame som passer game state manager interface"""

    def __init__(self, config):
        self.config = config
        self.game = CatchGame(self.config)
        self.state_cache = {}  

    def generate_initial_state(self):
        """generer initial game state"""
        return self.game.reset()

    def get_legal_actions(self, state):
        """returnerer alle lovlige actions for en state"""
        temp_game = self._create_temp_game(state)
        return temp_game.get_legal_actions()

    def get_next_state_and_reward(self, state, action):
        """returnerer neste state og reward gitt state og action"""
        cache_key = (state.tobytes(), action)
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]

        temp_game = self._create_temp_game(state)

        next_state, reward, done = temp_game.step(action)

        self.state_cache[cache_key] = (next_state, reward, done)

        return next_state, reward, done

    def is_terminal_state(self, state):
        """sjekk om state er terminal"""
        temp_game = self._create_temp_game(state)
        return temp_game.game_over

    def evaluate_state(self, state):
        """enkel heuristisk evaluering for en state"""
        temp_game = self._create_temp_game(state)
        if temp_game.game_over:
            return -10

        fruit_col = temp_game.fruit_pos[1]
        paddle_center = temp_game.paddle_pos + temp_game.paddle_size // 2
        x_distance_to_fruit = abs(fruit_col - paddle_center)
        fruit_distance_to_ground = temp_game.height - temp_game.fruit_pos[0] - 1

        if x_distance_to_fruit <= fruit_distance_to_ground:
            return 1
        else:
            return -1

    def _create_temp_game(self, state):
        """lag et midlertidig game med gitt state"""
        temp_game = CatchGame(self.config)

        paddle_indices = np.where(state[-1] == 1)[0]
        if len(paddle_indices) > 0:
            temp_game.paddle_pos = paddle_indices[0]

        fruit_pos = np.where(state == 2)
        if len(fruit_pos[0]) > 0:
            temp_game.fruit_pos = [fruit_pos[0][0], fruit_pos[1][0]]
        else:
            temp_game.fruit_pos = [0, np.random.randint(0, temp_game.width)]

        temp_game.grid = state.copy()

        return temp_game

    def state_to_tensor(self, state=None):
        """
        Convert a game state to a PyTorch tensor suitable for neural network input.

        Args:
            state: Game state to convert. If None, use current state.

        Returns:
            torch.Tensor: Tensor ready for neural network input
        """
        if state is None:
            state = self.game.get_state()

        observation_channels = self.game.state_to_observation(state)

        tensor_observation = torch.tensor(observation_channels, dtype=torch.float32)

        if len(tensor_observation.shape) == 3:  
            tensor_observation = tensor_observation.unsqueeze(0)  

        return tensor_observation
