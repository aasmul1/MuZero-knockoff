import numpy as np

class CatchGame:
    def __init__(self, grid_width=10, grid_height=10, paddle_size=3):
        """Initialize the game state."""
        self.width = grid_width
        self.height = grid_height
        self.paddle_size = paddle_size
        self.reset()
        
    def reset(self):
        """Reset the game to initial state."""
        # Empty grid
        self.grid = np.zeros((self.height, self.width))
        
        # Place paddle at bottom center
        self.paddle_pos = self.width // 2 - self.paddle_size // 2
        self.update_paddle()
        
        # Initialize fruit at random position at top
        self.fruit_pos = [0, np.random.randint(0, self.width)]
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2  # 2 represents fruit
        
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_state()
    
    def update_paddle(self):
        """Update paddle position on grid."""
        # Clear bottom row
        self.grid[self.height-1, :] = 0
        
        # Place paddle
        for i in range(self.paddle_size):
            if 0 <= self.paddle_pos + i < self.width:
                self.grid[self.height-1, self.paddle_pos + i] = 1  # 1 represents paddle
    
    def get_state(self):
        """Return current state representation for agent."""
        return self.grid.copy()
    
    def get_legal_actions(self):
        """Return list of legal actions."""
        actions = []
        if self.paddle_pos > 0:
            actions.append(0)  # Move left
        actions.append(1)      # Stay
        if self.paddle_pos + self.paddle_size < self.width:
            actions.append(2)  # Move right
        return actions
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)."""
        # Move paddle based on action
        if action == 0 and self.paddle_pos > 0:  # Left
            self.paddle_pos -= 1
        elif action == 2 and self.paddle_pos + self.paddle_size < self.width:  # Right
            self.paddle_pos += 1
        # Action 1 is stay (no movement)
        
        self.update_paddle()
        
        # Move fruit down
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 0
        self.fruit_pos[0] += 1
        
        # Check if fruit is at bottom row
        reward = 0
        if self.fruit_pos[0] == self.height - 1:
            # Check if fruit is caught by paddle
            if self.paddle_pos <= self.fruit_pos[1] < self.paddle_pos + self.paddle_size:
                reward = 1
                self.score += 1
            else:
                reward = -1
                self.game_over = True
                
            # Spawn new fruit at top if game continues
            if not self.game_over:
                self.fruit_pos = [0, np.random.randint(0, self.width)]
        
        # Place fruit on grid if game continues
        if not self.game_over:
            self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2
        
        self.steps += 1
        
        # Optional: End game after certain number of steps
        if self.steps >= 100:
            self.game_over = True
            
        return self.get_state(), reward, self.game_over
    
    def render(self, mode='human'):
        """Render the current game state."""
        if mode == 'human':
            for row in self.grid:
                line = ""
                for cell in row:
                    if cell == 0:
                        line += "⬛"  # Empty
                    elif cell == 1:
                        line += "🟦"  # Paddle
                    elif cell == 2:
                        line += "🍎"  # Fruit
                print(line)
            print(f"Score: {self.score}")
            print("-" * self.width)
        return self.grid
    
    def manual_play(self):
        """Allow manual play using keyboard input."""
        from pynput import keyboard
        
        def on_press(key):
            try:
                if key == keyboard.Key.left:
                    self.step(0)  # Left
                elif key == keyboard.Key.right:
                    self.step(2)  # Right
                elif key == keyboard.Key.space:
                    self.step(1)  # Stay
                
                # Clear screen and render
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                self.render()
                
                if self.game_over:
                    print("Game Over! Final score:", self.score)
                    return False  # Stop listener
            except:
                pass
                
        # Start listening for key presses
        with keyboard.Listener(on_press=on_press) as listener:
            self.render()  # Initial render
            listener.join()
    
    def clone(self):
        """Create a deep copy of the game state for MCTS."""
        new_game = CatchGame(self.width, self.height, self.paddle_size)
        new_game.grid = self.grid.copy()
        new_game.paddle_pos = self.paddle_pos
        new_game.fruit_pos = self.fruit_pos.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.steps = self.steps
        return new_game

    def get_action_space_size(self):
        """Return the size of the action space."""
        return 3  # Left, Stay, Right

    def get_state_size(self):
        """Return the dimensions of the state."""
        return (self.height, self.width)

    def state_to_observation(self, state=None):
        """Convert state to observation format for neural networks."""
        if state is None:
            state = self.get_state()
        
        # For simple representation: same as state
        # For MuZero: you might want to use one-hot encoding
        # e.g., channels: [empty cells, paddle positions, fruit positions]
        channels = np.zeros((3, self.height, self.width))
        
        channels[0] = (state == 0).astype(np.float32)  # Empty
        channels[1] = (state == 1).astype(np.float32)  # Paddle
        channels[2] = (state == 2).astype(np.float32)  # Fruit
        
        return channels

    