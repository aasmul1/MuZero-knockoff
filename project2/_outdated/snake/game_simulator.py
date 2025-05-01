import numpy as np
import random

class SnakeGameAI:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        self.done = False
        self.score = 0
        self.direction = "RIGHT"  
        mid = self.grid_size // 2
        # Initialize snake as a list of coordinates (head is first element)
        self.snake = [(mid, mid), (mid, mid-1), (mid, mid-2)]
        self.spawn_fruit()
        self.update_grid()
        return self.get_state()
    
    def spawn_fruit(self):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if (x, y) not in self.snake:
                self.fruit = (x, y)
                break
            
    def is_collision(self, cell):
        # Check if the cell is out of bounds or hits the snake body
        return (cell[0] in (-1, self.grid_size) or
                cell[1] in (-1, self.grid_size) or
                cell in self.snake)
            
    def update_grid(self):
        self.game = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for cell in self.snake:
            self.game[cell[0], cell[1]] = 1
        self.game[self.fruit[0], self.fruit[1]] = 2
        
    def get_state(self):
        return self.game.copy()
    
    # def print_grid(self):
    #     # Print the grid with borders. Snake cells will show "S", fruit "F", and empty cells as blank.
    #     cell_width = 3
    #     num_rows, num_cols = self.game.shape
    #     total_width = num_cols * cell_width + 2
    #     print("+" + "-" * (total_width - 2) + "+")
    #     for row in self.game:
    #         row_str = "|"
    #         for cell in row:
    #             if cell == 0:
    #                 row_str += "   "
    #             elif cell == 1:
    #                 row_str += " S "
    #             elif cell == 2:
    #                 row_str += " F "
    #         row_str += "|"
    #         print(row_str)
    #     print("+" + "-" * (total_width - 2) + "+")
    #     print("Score:", self.score)
    
    def move(self, direction):
        if direction == [1,0,0]:
            new_head = (self.snake[0][0] + 1, self.snake[0][1])
        elif direction == [0,1,0]:
            new_head = (self.snake[0][0], self.snake[0][1] + 1)
        elif direction == [0,0,1]:
            new_head = (self.snake[0][0] - 1, self.snake[0][1])
            
        
        
    def step(self, action):
        """
        Take a step in the game given an action.
        action: one of "UP", "DOWN", "LEFT", "RIGHT"
        Returns: (state, reward, done)
        """
        self.move(action)   
        # Prevent reversing direction if snake length > 1.
        opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        if len(self.snake) > 1 and action == opposites[self.direction]:
            action = self.direction  # ignore the invalid (reverse) move
        self.direction = action
        
        head_x, head_y = self.snake[0]
        if self.direction == "UP":
            new_head = (head_x - 1, head_y)
        elif self.direction == "DOWN":
            new_head = (head_x + 1, head_y)
        elif self.direction == "LEFT":
            new_head = (head_x, head_y - 1)
        elif self.direction == "RIGHT":
            new_head = (head_x, head_y + 1)
        
        # Check for collisions: hitting walls or the snake itself
        if self.is_collision(new_head):
            self.done = True
            reward = -10  # Penalty for collision
            return self.get_state(), reward, self.done
        
        # Insert the new head at the beginning of the snake list
        self.snake.insert(0, new_head)
        
        # Check if fruit is eaten
        if new_head == self.fruit:
            self.score += 1
            reward = 10  # Reward for eating fruit
            self.spawn_fruit()
        else:
            # Remove the tail (move forward without growing)
            self.snake.pop()
            reward = 0
        
        self.update_grid()
        return self.score, reward, self.done

