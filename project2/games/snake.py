import numpy as np
import random

class SnakeGame:
    def __init__(self, grid_size=10, grow_on_food=True):
        self.grid_size = grid_size
        self.grow_on_food = grow_on_food  # flag for å kontrollere slangevekst
        self.reset()
        
    def reset(self):
        self.done = False
        self.score = 0
        self.direction = "RIGHT"  
        mid = self.grid_size // 2
        # initialiser slange som en liste med koordinater (hodet som første element)
        self.snake = [(mid, mid)]
        if self.grow_on_food:
            self.snake.extend([(mid-1, mid), (mid-2, mid)])
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
    
    def get_legal_actions(self):
        """returner liste med lovlige actions: 0=venstre, 1=opp, 2=høyre, 3=ned"""
        actions = []
        # forhindre reversering av retning
        if self.direction != "RIGHT":
            actions.append(0)  # venstre
        if self.direction != "DOWN":
            actions.append(1)  # opp
        if self.direction != "LEFT":
            actions.append(2)  # høyre
        if self.direction != "UP":
            actions.append(3)  # ned
        return actions
            
    def is_collision(self, point):
        """sjekk om cellen er utenfor brettet eller treffer slangekroppen"""
        x, y = point
        return (x < 0 or x >= self.grid_size or
                y < 0 or y >= self.grid_size or
                point in self.snake[1:])
            
    def update_grid(self):
        """oppdater rutenett basert på nåværende state"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for cell in self.snake:
            self.grid[cell] = 1
        self.grid[self.fruit] = 2
        
    def get_state(self):
        """returner state"""
        return self.grid.copy()
    
    def step(self, action):
        """utfør action (0=venstre, 1=opp, 2=høyre, 3=ned) og returner (next_state, reward, done)"""
        # konverter handling til retning
        if action == 0:
            self.direction = "LEFT"
        elif action == 1:
            self.direction = "UP"
        elif action == 2:
            self.direction = "RIGHT"
        elif action == 3:
            self.direction = "DOWN"
        
        # få hodeposisjon
        head_x, head_y = self.snake[0]
        
        # flytt hode i henhold til retning
        if self.direction == "LEFT":
            new_head = (head_x, head_y - 1)
        elif self.direction == "RIGHT":
            new_head = (head_x, head_y + 1)
        elif self.direction == "UP":
            new_head = (head_x - 1, head_y)
        elif self.direction == "DOWN":
            new_head = (head_x + 1, head_y)
        
        # sjekk for kollisjon
        if self.is_collision(new_head):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done
        
        # legg til nytt hode
        self.snake.insert(0, new_head)
        
        # sjekk om frukt er spist
        if new_head == self.fruit:
            self.score += 1
            reward = 10
            self.spawn_fruit()
            # Bare behold halen hvis grow_on_food er True
            if not self.grow_on_food:
                self.snake.pop()
        else:
            # Alltid fjern halen hvis ikke frukt er spist
            self.snake.pop()
            reward = 0
        
        self.update_grid()
        return self.get_state(), reward, self.done
    
    def render(self, mode='human'):
        """render nåværende gamestate"""
        if mode == 'human':
            # tøm konsollen
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # lag visuell representasjon
            for row in range(self.grid_size):
                line = ""
                for col in range(self.grid_size):
                    if (row, col) == self.snake[0]:
                        line += "🟢"  # slangehode
                    elif (row, col) in self.snake[1:]:
                        line += "🟩"  # slangekropp
                    elif (row, col) == self.fruit:
                        line += "🍎"  # frukt
                    else:
                        line += "⬛"  # tomt
                print(line)
            print(f"Score: {self.score}")
            print("-" * (self.grid_size * 2))
            
        return self.grid