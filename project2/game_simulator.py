import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np

class GameSim:
    
    def __init__(self, i, game, max_rewards):
        self.snake_index = i #array index
        self.game = game #array
        self.max_rewards = max_rewards
        self.done = False
        
    def step(self, action):
        
        if self.done:
            return self.game, 0, self.done
        
        new_index = self.snake_index + action
        if new_index < 0 or new_index >= len(self.game):
            reward = -3
            self.done = True
            return self.game, reward, self.done

        self.snake_index = new_index
        cell_value = self.game[self.snake_index]
        reward = cell_value
        self.game[self.snake_index] = 9
        self.game[self.snake_index-action] = 0
        

        return self.game, reward, self.done
        
        
    def print_grid(self):
        str = "|"
        for i in range(0, len(self.game)):
            str += f"{self.game[i]} |"
        print(str)
        

# Assume actions: 0 means move left (-1), 1 means move right (+1)
def train_q_learning(num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
    # For simplicity, treat each snake position (index) as a state.
    num_states = 9  # for your 1D grid with 9 cells
    q_table = np.zeros((num_states, 2))  # 2 actions: left and right
    rewards_per_episode = []
    
    for episode in range(num_episodes):
        # Create a new game copy for each episode
        game_copy = [1, 0, 0, 0, 0, 0, 9, 1, 0]
        # Let's assume starting index 3 as in your example.
        env = GameSim(6, game_copy, 2)
        total_reward = 0
        
        while not env.done:
            state = env.snake_index
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_choice = random.choice([0, 1])
            else:
                action_choice = np.argmax(q_table[state])
            action = -1 if action_choice == 0 else 1
            
            # Take action and get feedback from the environment
            _, reward, done = env.step(action)
            next_state = env.snake_index
            
            # Update Q-table
            q_table[state, action_choice] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action_choice])
            
            total_reward += reward
            if total_reward == env.max_rewards:
                break
        
        epsilon *= epsilon_decay
        rewards_per_episode.append(total_reward)
        
        
    
    return q_table, rewards_per_episode

if __name__ == "__main__":
    q_table, rewards = train_q_learning()
    print("Trained Q-table:")
    print(q_table)

    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # Optionally, plot a running average to smooth the curve
    window = 20  # window size for moving average
    running_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(running_avg, label=f"Running Average (window={window})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve (Smoothed)")
    plt.legend()
    plt.show()
            
# if __name__ == "__main__":
#     game = [0,0,-1,1,0,0,-1,0,0]
#     game_sim = GameSim(3, game)
    
#     for action in [1, 1, 1, 1, 1, 1]:
#         state, reward, done = game_sim.step(action)
#         game_sim.print_grid()
#         print(f"Action: {action} | Reward: {reward} | Done: {done}")
#         if done:
#             print("Game over!")
#             break