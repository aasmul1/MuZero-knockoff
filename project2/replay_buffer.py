import numpy as np
import torch
import logging


class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer_size = getattr(config, "REPLAY_BUFFER_SIZE", 10000)  # Default if not specified
        self.batch_size = getattr(config, "BATCH_SIZE", 128)  # Default if not specified
        self.num_unroll_steps = getattr(config, "NUM_UNROLL_STEPS", 5)  # For n-step returns
        self.td_steps = getattr(config, "TD_STEPS", 10)  # For bootstrapping
        self.discount = getattr(config, "DISCOUNT", 0.997)  # Discount factor for rewards
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        
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
        if not self.buffer:
            raise ValueError("Buffer is empty, cannot sample batch.")
            
        effective_batch_size = min(self.batch_size, len(self.buffer) * 10)
        if effective_batch_size < 1:
            effective_batch_size = 1
            
        games = np.random.choice(self.buffer, size=effective_batch_size // 2)
        game_positions = []
        
        for game in games:
            if len(game.observations) > 0:
                pos = np.random.randint(0, len(game.observations))
                game_positions.append((game, pos))
                
        if not game_positions:
            raise ValueError("No valid game positions found.")
        
        batch = {
            'observations': [],
            'actions': [],
            'target_value': [],
            'target_reward': [],
            'target_policy': [] 
        }
        
        for game, position in game_positions:
            targets = self.make_target(game, position)
            if targets:
                for key in batch.keys():
                    if key in targets and targets[key] is not None:
                        batch[key].append(targets[key])
        
        try:
            if batch['observations']:
                if isinstance(batch['observations'][0], torch.Tensor):
                    batch['observations'] = torch.stack(batch['observations']).float()
                else:
                    batch['observations'] = torch.tensor(batch['observations'], dtype=torch.float32)
            else:
                batch['observations'] = torch.tensor([], dtype=torch.float32)
                
            if batch['actions']:
                action_lists = []
                for action_seq in batch['actions']:
                    action_numbers = []
                    for a in action_seq:
                        if hasattr(a, 'number'):
                            action_numbers.append(a.number)
                        else:
                            action_numbers.append(a)
                    action_lists.append(action_numbers)
                if action_lists:
                    max_len = max(len(seq) for seq in action_lists)
                    padded_actions = [seq + [0] * (max_len - len(seq)) for seq in action_lists]
                    batch['actions'] = torch.tensor(padded_actions, dtype=torch.long)
                else:
                    batch['actions'] = torch.tensor([], dtype=torch.long)
            else:
                batch['actions'] = torch.tensor([], dtype=torch.long)
                
            if batch['target_value']:
                max_len = max(len(val_list) for val_list in batch['target_value'])
                padded_values = []
                for val_list in batch['target_value']:
                    padded_values.append(val_list + [0.0] * (max_len - len(val_list)))
                batch['target_value'] = torch.tensor(padded_values, dtype=torch.float32)
            else:
                batch['target_value'] = torch.tensor([], dtype=torch.float32)
                
            if batch['target_reward']:
                max_len = max(len(rew_list) for rew_list in batch['target_reward'])
                padded_rewards = []
                for rew_list in batch['target_reward']:
                    padded_rewards.append(rew_list + [0.0] * (max_len - len(rew_list)))
                batch['target_reward'] = torch.tensor(padded_rewards, dtype=torch.float32)
            else:
                batch['target_reward'] = torch.tensor([], dtype=torch.float32)
                
            if batch['target_policy']:
                policy_tensors = []
                for policy_seq in batch['target_policy']:
                    seq_tensors = []
                    for policy_dict in policy_seq:
                        policy_vec = [0.0] * self.config.ACTION_SPACE
                        for a, prob in policy_dict.items():
                            a_idx = a.number if hasattr(a, 'number') else a
                            policy_vec[a_idx] = float(prob)
                        seq_tensors.append(policy_vec)
                    
                    max_seq_len = max(len(p) for p in batch['target_policy'])
                    if len(seq_tensors) < max_seq_len:
                        for _ in range(max_seq_len - len(seq_tensors)):
                            seq_tensors.append([0.0] * self.config.ACTION_SPACE)
                            
                    policy_tensors.append(seq_tensors)
                    
                batch['target_policy'] = torch.tensor(policy_tensors, dtype=torch.float32)
            else:
                batch['target_policy'] = torch.tensor([], dtype=torch.float32)
            
            if len(batch['observations']) == 0:
                self.logger.warning("No observations in batch!")
            
            self.logger.debug(f"Batch shapes - observations: {batch['observations'].shape}, "
                             f"actions: {batch['actions'].shape}, "
                             f"target_value: {batch['target_value'].shape}, "
                             f"target_reward: {batch['target_reward'].shape}, "
                             f"target_policy: {batch['target_policy'].shape}")
                
        except Exception as e:
            self.logger.error(f"Error creating batch tensors: {str(e)}")
            raise
        
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
        if position >= len(game.observations):
            return None
            
        try:
            targets = {
                'observations': game.observations[position],
                'actions': [],
                'target_value': [],
                'target_reward': [],
                'target_policy': [] 
            }
            
            if position < len(game.search_policies):
                targets['target_policy'].append(game.search_policies[position])
            else:
                action_space = getattr(self.config, "ACTION_SPACE", 3)
                uniform_policy = {a: 1.0/action_space for a in range(action_space)}
                targets['target_policy'].append(uniform_policy)
            
            value = self.compute_target_value(game, position)
            targets['target_value'].append(value)
            
            max_steps = min(position + self.num_unroll_steps, len(game.actions))
            for current_index in range(position, max_steps):
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
        except Exception as e:
            self.logger.error(f"Error creating target at position {position}: {e}")
            return None
    
    def compute_target_value(self, game, position):
        """
        Compute the target value using TD(n) with bootstrap.
        
        Args:
            game: A game trajectory
            position: Position in the game
            
        Returns:
            Target value (float)
        """
        bootstrap_index = min(len(game.rewards) - 1, position + self.td_steps)
        
        bootstrap_value = game.values[bootstrap_index] if bootstrap_index < len(game.values) else 0
        
        value = bootstrap_value * self.discount**self.td_steps
        
        for i in range(position, bootstrap_index):
            if i+1 < len(game.rewards):  
                value += game.rewards[i+1] * self.discount**(i - position)
        
        return value