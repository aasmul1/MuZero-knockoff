import numpy as np
from numpy.random import default_rng


class CatchGame:
    def __init__(self, config):
        self.config = config
        self.width = self.config.CATCH_GRID_WIDTH
        self.height = self.config.CATCH_GRID_HEIGHT
        self.paddle_size = self.config.PADDLE_SIZE
        # Seed explicitly for reproducibility
        seed = getattr(config, 'SEED', None)
        self.rng = default_rng(seed)
        self.reset()

    def reset(self):
        """Resett spillet til initial state"""
        self.grid = np.zeros((self.height, self.width))

        # Paddle init midtstilt, begrenset innenfor grid
        self.paddle_pos = self.width // 2 - self.paddle_size // 2
        assert 0 <= self.paddle_pos <= self.width - self.paddle_size, \
            f"Paddle start pos out of bounds: {self.paddle_pos}"
        self.update_paddle()

        # Frukt spawn på topprad, kolonne innenfor [0, width)
        col = self.rng.integers(0, self.width)
        assert 0 <= col < self.width, f"Fruit spawn out of bounds: {col}"
        self.fruit_pos = [0, col]
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2

        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_state()

    def update_paddle(self):
        """Oppdater paddle posisjon på grid"""
        # Tøm nederste rad
        self.grid[self.height - 1, :] = 0

        # Plasser paddle innenfor grenser
        assert 0 <= self.paddle_pos <= self.width - self.paddle_size, \
            f"Paddle pos out of bounds: {self.paddle_pos}"
        for i in range(self.paddle_size):
            self.grid[self.height - 1, self.paddle_pos + i] = 1  # 1 representerer paddle

    def get_state(self):
        """Returner nåværende state representasjon for agent"""
        return self.grid.copy()

    def get_legal_actions(self):
        """Returner liste av lovlige actions"""
        actions = []
        if self.paddle_pos > 0:
            actions.append(0)  # move left
        actions.append(1)  # stay
        if self.paddle_pos + self.paddle_size < self.width:
            actions.append(2)  # move right
        return actions

    def step(self, action):
        """Utfør action og returner (next_state, reward, done)"""
        if action == 0 and self.paddle_pos > 0:
            self.paddle_pos -= 1
        elif action == 2 and self.paddle_pos + self.paddle_size < self.width:
            self.paddle_pos += 1

        self.update_paddle()

        # Flytt frukt nedover
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 0
        self.fruit_pos[0] += 1

        reward = 0
        if self.fruit_pos[0] == self.height - 1:
            if self.paddle_pos <= self.fruit_pos[1] < self.paddle_pos + self.paddle_size:
                reward = 1
                self.score += 1
            else:
                reward = -1
                self.game_over = True

            if not self.game_over:
                # Spawn ny frukt
                col = self.rng.integers(0, self.width)
                assert 0 <= col < self.width, f"Fruit spawn out of bounds: {col}"
                self.fruit_pos = [0, col]

        if not self.game_over:
            self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2

        self.steps += 1
        if self.steps >= 100:
            self.game_over = True

        return self.get_state(), reward, self.game_over

    def render(self, mode='human'):
        """Render nåværende game state"""
        if mode == 'human':
            for row in self.grid:
                line = ''
                for cell in row:
                    if cell == 0:
                        line += '⬛'
                    elif cell == 1:
                        line += '🟦'
                    elif cell == 2:
                        line += '🍎'
                print(line)
            print(f"Score: {self.score}")
            print('-' * self.width)
        return self.grid

    def clone(self):
        """Lag en deep copy av game state for MCTS, inkludert RNG-state"""
        new_game = CatchGame(self.config)
        # Kopier RNG-state for determinisme
        state = self.rng.bit_generator.state
        new_game.rng = default_rng()
        new_game.rng.bit_generator.state = state.copy()
        new_game.grid = self.grid.copy()
        new_game.paddle_pos = self.paddle_pos
        new_game.fruit_pos = self.fruit_pos.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.steps = self.steps
        return new_game

    def get_action_space_size(self):
        return 3

    def get_state_size(self):
        return (self.height, self.width)

    def state_to_observation(self, state=None):
        if state is None:
            state = self.get_state()
        channels = np.zeros((3, self.height, self.width))
        channels[0] = (state == 0).astype(np.float32)
        channels[1] = (state == 1).astype(np.float32)
        channels[2] = (state == 2).astype(np.float32)
        return channels
