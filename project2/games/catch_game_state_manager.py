import numpy as np
import torch

from project2.games.generic_game_state_manager import GameStateManager


class CatchGameStateManager(GameStateManager):
    """Adapter klasse for CatchGame som passer game state manager interface"""

    def __init__(self, config):
        from project2.games.catch import CatchGame
        self.config = config
        self.game = CatchGame(self.config)
        self.state_cache = {}
        self.max_total_rewards = config.MAX_TOTAL_REWARDS
        self.total_reward = 0

    def generate_initial_state(self):
        """Generer initial game state og nullstill cache"""
        self.state_cache.clear()
        return self.game.reset()

    def get_legal_actions(self, state):
        temp_game = self._create_temp_game(state)
        return temp_game.get_legal_actions()

    def get_next_state_and_reward(self, state, action):
        cache_key = (state.tobytes(), action)
        if cache_key in self.state_cache:
            next_state, reward, done = self.state_cache[cache_key]
            # oppdater total_reward og sjekk cutoff
            self.total_reward += reward
            if self.total_reward >= self.max_total_rewards:
                done = True
            return next_state, reward, done

        temp_game = self._create_temp_game(state)
        next_state, reward, done = temp_game.step(action)
        self.total_reward += reward
        if self.total_reward >= self.max_total_rewards:
            done = True
        self.state_cache[cache_key] = (next_state, reward, done)
        return next_state, reward, done

    def is_terminal_state(self, state):
        temp_game = self._create_temp_game(state)
        return temp_game.game_over

    def evaluate_state(self, state):
        temp_game = self._create_temp_game(state)
        if temp_game.game_over:
            return -10
        fruit_col = temp_game.fruit_pos[1]
        paddle_center = temp_game.paddle_pos + temp_game.paddle_size // 2
        x_distance_to_fruit = abs(fruit_col - paddle_center)
        fruit_distance_to_ground = temp_game.height - temp_game.fruit_pos[0] - 1
        return 1 if x_distance_to_fruit <= fruit_distance_to_ground else -1

    def _create_temp_game(self, state):
        from project2.games.catch import CatchGame
        temp_game = CatchGame(self.config)
        # Kopier paddle-pos fra state
        paddle_indices = np.where(state[-1] == 1)[0]
        if len(paddle_indices) > 0:
            temp_game.paddle_pos = int(paddle_indices[0])
        # Kopier fruit-pos fra state eller bruk temp_game.rng
        fruit_pos = np.where(state == 2)
        if len(fruit_pos[0]) > 0:
            temp_game.fruit_pos = [int(fruit_pos[0][0]), int(fruit_pos[1][0])]
        else:
            col = temp_game.rng.integers(0, temp_game.width)
            assert 0 <= col < temp_game.width, f"Fruit spawn out of bounds: {col}"
            temp_game.fruit_pos = [0, col]
        temp_game.grid = state.copy()
        return temp_game

    def state_to_tensor(self, state=None):
        if state is None:
            state = self.game.get_state()
        observation_channels = self.game.state_to_observation(state)
        tensor_observation = torch.tensor(observation_channels, dtype=torch.float32)
        if tensor_observation.ndim == 3:
            tensor_observation = tensor_observation.unsqueeze(0)
        return tensor_observation
