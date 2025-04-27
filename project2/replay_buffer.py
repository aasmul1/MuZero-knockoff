import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer_size = getattr(config, "REPLAY_BUFFER_SIZE", 10000)  # Default if not specified
        self.batch_size = getattr(config, "BATCH_SIZE", 128)  # Default if not specified
        self.num_unroll_steps = getattr(config, "NUM_UNROLL_STEPS", 5)  # For n-step returns
        self.td_steps = getattr(config, "TD_STEPS", 10)  # For bootstrapping
        self.discount = getattr(config, "DISCOUNT", 0.997)  # Discount factor for rewards
        
        self.buffer = []

    def save_game(self, game):
        """Add a game trajectory to the buffer, removing oldest if buffer is full."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        """
        Sample a batch of training examples from the buffer.
        
        Returns:
            Dictionary with tensors ready for training.
        """
        # Check if we have any games in the buffer
        if not self.buffer:
            raise ValueError("Buffer is empty, cannot sample batch.")
            
        # Make sure batch size doesn't exceed available data
        effective_batch_size = min(self.batch_size, len(self.buffer) * 10)
        
        # Sample games and positions
        games = np.random.choice(self.buffer, size=effective_batch_size // 2)
        game_positions = []
        
        for game in games:
            if len(game.observations) > 0:
                pos = np.random.randint(0, len(game.observations))
                game_positions.append((game, pos))
                
        if not game_positions:
            raise ValueError("No valid game positions found.")
        
        # Initialize batch dictionary
        batch = {
            'observations': [],
            'actions': [],
            'target_value': [],
            'target_reward': [],
            'target_policy': [] 
        }
        
        # Fill the batch with data from sampled positions
        for game, position in game_positions:
            targets = self.make_target(game, position)
            for key in batch.keys():
                if key in targets and targets[key] is not None:
                    batch[key].append(targets[key])
        
       
        return batch
        
    def make_target(self, game, position):
        """
        Create a target for training from a specific position in a game.
        
        Args:
            game: A game trajectory
            position: Position in the game to start from
            
        Returns:
            Training targets for this position
        """
        # Make sure position is valid
        if position >= len(game.observations):
            return None
            
        # Initialize target dictionary
        targets = {
            'observations': game.observations[position],
            'actions': [],
            'target_value': [],
            'target_reward': [],
            'target_policy': [] 
        }
        
        # Add initial policy and value
        if position < len(game.search_policies):
            targets['target_policy'].append(game.search_policies[position])
        else:
            # Create uniform policy if missing
            action_space = getattr(self.config, "ACTION_SPACE", 3)
            uniform_policy = {a: 1.0/action_space for a in range(action_space)}
            targets['target_policy'].append(uniform_policy)
        
        # Calculate bootstrapped value target
        value = self.compute_target_value(game, position)
        targets['target_value'].append(value)
        
        # Add unroll steps
        for current_index in range(position, min(position + self.num_unroll_steps, len(game.actions))):
            action = game.actions[current_index]
            targets['actions'].append(action)
            
            # Get policy and value for next position
            next_index = current_index + 1
            if next_index < len(game.search_policies):
                targets['target_policy'].append(game.search_policies[next_index])
                value = self.compute_target_value(game, next_index)
                targets['target_value'].append(value)
                
            # Get reward
            if next_index < len(game.rewards):
                targets['target_reward'].append(game.rewards[next_index])
                
        return targets
    
    def compute_target_value(self, game, position):
        """
        Compute the target value using TD(n) with bootstrap.
        
        Args:
            game: A game trajectory
            position: Position in the game
            
        Returns:
            Target value (float)
        """
        # Determine bootstrap index
        bootstrap_index = min(len(game.rewards) - 1, position + self.td_steps)
        
        # Get bootstrap value (use 0 if beyond game length)
        bootstrap_value = game.values[bootstrap_index] if bootstrap_index < len(game.values) else 0
        
        # Apply discount to bootstrap value
        value = bootstrap_value * self.discount**self.td_steps
        
        # Sum immediate rewards with discount
        for i in range(position, bootstrap_index):
            if i+1 < len(game.rewards):  # Make sure we don't go out of bounds
                value += game.rewards[i+1] * self.discount**(i - position)
        
        return value