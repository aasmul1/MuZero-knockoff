import matplotlib.pyplot as plt
import numpy as np
import time

class GameVisualizer:
    """klasse for å visualisere spilltilstander og training progress"""
    
    def __init__(self, game_type='snake'):
        self.game_type = game_type
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'losses': []
        }
        
    def display_game_state(self, state, score=0, mode='console'):
        """vis en enkelt spilltilstand"""
        if mode == 'console':
            # konsoll-visualisering
            for row in state:
                line = ""
                for cell in row:
                    if cell == 0:
                        line += "⬛"  # tomt
                    elif cell == 1:
                        if self.game_type == 'snake':
                            line += "🟩"  # slange
                        else:  # catch
                            line += "🟦"  # racket
                    elif cell == 2:
                        if self.game_type == 'snake':
                            line += "🍎"  # frukt
                        else:  # catch
                            line += "🍎"  # frukt
                print(line)
            print(f"Score: {score}")
            print("-" * (len(state) * 2))
        elif mode == 'plot':
            # matplotlib-visualisering
            plt.figure(figsize=(6, 6))
            plt.imshow(state, cmap='viridis')
            plt.title(f"Game State - Score: {score}")
            plt.axis('off')
            plt.show()
    
    def record_training_stats(self, episode, reward, loss=None):
        """lagre training stats for visualisering"""
        self.training_stats['episodes'].append(episode)
        self.training_stats['rewards'].append(reward)
        if loss is not None:
            self.training_stats['losses'].append(loss)
    
    def plot_training_progress(self):
        """plot training"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # plott belønninger
        ax1.plot(self.training_stats['episodes'], self.training_stats['rewards'], 'b-')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.grid(True)
        
        # plott tap hvis tilgjengelig
        if len(self.training_stats['losses']) > 0:
            ax2.plot(self.training_stats['episodes'], self.training_stats['losses'], 'r-')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()