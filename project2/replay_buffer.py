import numpy as np


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.REPLAY_BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        self.num_unroll_steps = config.NUM_UNROLL_STEPS
        self.td_steps = config.TD_STEPS
        
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) + 1 > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = np.random.choice(self.buffer, size=self.bathc_size // 2)
        game_positions = [(game, np.random.randint(0, len(game))) for game in games]
        batch = {
            'observations': [],
            'actions': [],
            'target_value': [],
            'target_reward': [],
            'target_policy': [] 
        }
        
        for game, position in game_positions:
            return
        
    def make_target(self, game, position):
        targets = {
            'observations': game.observation[position],
            'actions': [],
            'target_value': [],
            'target_reward': [],
            'target_policy': [] 
        }
        targets['target_policy'].append(game.search_policies[position])
        
        value = self.compute_target_value(game, position)
        targets['target_value'].appeng(value)
        
        for current_index in range(position, min((position + self.num_unroll_steps), len(game.actions))):
            action = game.actions[current_index]
            targets['actions'].append(action)
            
            next_index = current_index + 1
            if next_index < len(game.search_policies):
                targets['target_policy'].append(game.search_policies[next_index])
                value = self.compute_target_value(game, next_index)
                targets['target_value'].append(value)
                
            if next_index < len(game.rewards):
                targets['target_reward'].append(game.rewards[next_index])
                
        return targets
    
    def compute_target_value(self, game, position):
        bootstrap_index = min((len(game.rewards) - 1), position + self.td_steps)
        
        bootstrap_value = game.values[bootstrap_index] if bootstrap_index < len(game.values) else 0
        value = bootstrap_value * self.config.DISCOUNT**self.td_steps
        
        for i in range(position, bootstrap_index):
            value += game.rewards[i+1] * self.config.DISCOUNT ** (i - position)
        
        return value