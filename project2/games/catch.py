import numpy as np

class CatchGame:
    def __init__(self, grid_width=10, grid_height=10, paddle_size=1):
        """initialiser game state"""
        self.width = grid_width
        self.height = grid_height
        self.paddle_size = paddle_size
        self.reset()
        
    def reset(self):
        """resett spillet til initial state"""
        # tomt grid
        self.grid = np.zeros((self.height, self.width))
        
        # plasser paddle nederst i midten
        self.paddle_pos = self.width // 2 - self.paddle_size // 2
        self.update_paddle()
        
        # initialiser frukt på tilfeldig posisjon på toppen
        self.fruit_pos = [0, np.random.randint(0, self.width)]
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2  # 2 representerer frukt
        
        self.score = 0
        self.game_over = False
        self.steps = 0
        return self.get_state()
    
    def update_paddle(self):
        """oppdater paddle posisjon på grid"""
        # tøm nederste rad
        self.grid[self.height-1, :] = 0
        
        # plasser paddle
        for i in range(self.paddle_size):
            if 0 <= self.paddle_pos + i < self.width:
                self.grid[self.height-1, self.paddle_pos + i] = 1  # 1 representerer paddle
    
    def get_state(self):
        """returner nåværende state representasjon for agent"""
        return self.grid.copy()
    
    def get_legal_actions(self):
        """returner liste av lovlige actions"""
        actions = []
        if self.paddle_pos > 0:
            actions.append(0)  # flytt venstre
        actions.append(1)      # stå stille
        if self.paddle_pos + self.paddle_size < self.width:
            actions.append(2)  # flytt høyre
        return actions
    
    def step(self, action):
        """utfør action og returner (next_state, reward, done)"""
        # flytt paddle basert på action
        if action == 0 and self.paddle_pos > 0:  # venstre
            self.paddle_pos -= 1
        elif action == 2 and self.paddle_pos + self.paddle_size < self.width:  # høyre
            self.paddle_pos += 1
        # action 1 er stå stille (ingen bevegelse)
        
        self.update_paddle()
        
        # flytt frukt nedover
        self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 0
        self.fruit_pos[0] += 1
        
        # sjekk om frukt er i nederste rad
        reward = 0
        if self.fruit_pos[0] == self.height - 1:
            # sjekk om frukt blir fanget av paddle
            if self.paddle_pos <= self.fruit_pos[1] < self.paddle_pos + self.paddle_size:
                reward = 1
                self.score += 1
            else:
                reward = -1
                self.game_over = True
                
            # spawn ny frukt på toppen hvis spillet fortsetter
            if not self.game_over:
                self.fruit_pos = [0, np.random.randint(0, self.width)]
        
        # plasser frukt på grid hvis spillet fortsetter
        if not self.game_over:
            self.grid[self.fruit_pos[0], self.fruit_pos[1]] = 2
        
        self.steps += 1
        
        # valgfritt: avslutt spill etter et visst antall steps
        if self.steps >= 100:
            self.game_over = True
            
        return self.get_state(), reward, self.game_over
    
    def render(self, mode='human'):
        """render nåværende game state"""
        if mode == 'human':
            for row in self.grid:
                line = ""
                for cell in row:
                    if cell == 0:
                        line += "⬛"  # tomt
                    elif cell == 1:
                        line += "🟦"  # paddle
                    elif cell == 2:
                        line += "🍎"  # frukt
                print(line)
            print(f"Score: {self.score}")
            print("-" * self.width)
        return self.grid
    
    def manual_play(self):
        """tillat manuell spilling med tastaturinput"""
        from pynput import keyboard
        
        def on_press(key):
            try:
                if key == keyboard.Key.left:
                    self.step(0)  # venstre
                elif key == keyboard.Key.right:
                    self.step(2)  # høyre
                elif key == keyboard.Key.space:
                    self.step(1)  # stå stille
                
                # tøm skjermen og render
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                self.render()
                
                if self.game_over:
                    print("Game Over! Final score:", self.score)
                    return False  # stopp lytter
            except:
                pass
                
        # start lytting etter tastetrykk
        with keyboard.Listener(on_press=on_press) as listener:
            self.render()  # initial render
            listener.join()
    
    def clone(self):
        """lag en deep copy av game state for MCTS"""
        new_game = CatchGame(self.width, self.height, self.paddle_size)
        new_game.grid = self.grid.copy()
        new_game.paddle_pos = self.paddle_pos
        new_game.fruit_pos = self.fruit_pos.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        new_game.steps = self.steps
        return new_game

    def get_action_space_size(self):
        """returner størrelsen på action space"""
        return 3  # venstre, stå stille, høyre

    def get_state_size(self):
        """returner dimensjonene til state"""
        return (self.height, self.width)

    def state_to_observation(self, state=None):
        """konverter state til observation format for neurale nettverk"""
        if state is None:
            state = self.get_state()
        
        # for enkel representasjon: samme som state
        # for MuZero: du vil kanskje bruke one-hot encoding
        # f.eks. channels: [tomme celler, paddle posisjoner, frukt posisjoner]
        channels = np.zeros((3, self.height, self.width))
        
        channels[0] = (state == 0).astype(np.float32)  # tomt
        channels[1] = (state == 1).astype(np.float32)  # paddle
        channels[2] = (state == 2).astype(np.float32)  # frukt
        
        return channels