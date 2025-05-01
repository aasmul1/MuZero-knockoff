import sys
import os
import torch
import numpy as np
import random
from collections import deque
import time

# Ensure the project2 module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from games.catch_game_state_manager import CatchGameStateManager
from project2.models.neural_net_manager import NeuralNetManager
import project2.core.config as config

def collect_self_play_data(net_manager, game_manager, num_games=10, max_steps=100):
    """Collect data from self-play games using the current network for action selection."""
    all_trajectories = []
    
    for game_num in range(num_games):
        # Reset game and get initial state
        state = game_manager.generate_initial_state()
        done = False
        step = 0
        
        # Initialize trajectory for this game
        observations = []
        actions = []
        rewards = []
        policies = []  # Store policies from the network (will be target policies for training)
        values = []    # Store value predictions
        
        while not done and step < max_steps:
            # Convert state to tensor
            observation_tensor = game_manager.state_to_tensor(state)
            observations.append(observation_tensor.squeeze(0).numpy())  # Remove batch dim for storage
            
            # Get network's prediction
            with torch.no_grad():
                network_output = net_manager.initial_inference(observation_tensor)
            
            # Get policy from network output
            policy = torch.softmax(network_output.policy_logits, dim=1).squeeze(0).numpy()
            value = network_output.value.item()
            
            # Get legal actions
            legal_actions = game_manager.get_legal_actions(state)
            
            # Choose action - initially with some exploration
            if random.random() < 0.2:  # Exploration rate
                action = random.choice(legal_actions)
            else:
                # Get best legal action according to policy
                legal_policy = np.zeros_like(policy)
                for a in legal_actions:
                    legal_policy[a] = policy[a]
                # If all legal actions have 0 probability, choose randomly
                if np.sum(legal_policy) == 0:
                    action = random.choice(legal_actions)
                else:
                    # Normalize and sample from legal policy
                    legal_policy = legal_policy / np.sum(legal_policy)
                    action = np.random.choice(len(legal_policy), p=legal_policy)
            
            # Store action and network outputs
            actions.append(action)
            policies.append(policy)
            values.append(value)
            
            # Take step in the game
            next_state, reward, done = game_manager.get_next_state_and_reward(state, action)
            rewards.append(reward)
            
            # Update for next step
            state = next_state
            step += 1
        
        # Store this game's trajectory
        trajectory = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'policies': np.array(policies),
            'values': np.array(values),
            'steps': step,
            'total_reward': sum(rewards)
        }
        all_trajectories.append(trajectory)
        
        print(f"Game {game_num+1}: Reward={trajectory['total_reward']}, Steps={step}")
    
    return all_trajectories

def prepare_batch(trajectories, batch_size=32, unroll_steps=5):
    """Prepare a batch of data for training from collected trajectories."""
    # Sample indices from all trajectories
    game_indices = []
    step_indices = []
    
    for game_idx, traj in enumerate(trajectories):
        # Ensure we have enough steps for unrolling
        max_start_idx = max(0, traj['steps'] - unroll_steps - 1)
        if max_start_idx > 0:
            for _ in range(min(10, max_start_idx)):  # Sample up to 10 positions from each game
                step_idx = random.randint(0, max_start_idx)
                game_indices.append(game_idx)
                step_indices.append(step_idx)
    
    # If we don't have enough data yet, just return None
    if len(game_indices) < batch_size:
        return None
    
    # Sample batch_size indices
    batch_indices = random.sample(range(len(game_indices)), min(batch_size, len(game_indices)))
    
    # Prepare batch data
    batch_observations = []
    batch_actions = []
    batch_target_policy = []
    batch_target_value = []
    batch_target_reward = []
    
    for idx in batch_indices:
        game_idx = game_indices[idx]
        step_idx = step_indices[idx]
        traj = trajectories[game_idx]
        
        # Get initial observation
        batch_observations.append(traj['observations'][step_idx])
        
        # Get actions for unrolling
        batch_actions.append(traj['actions'][step_idx:step_idx+unroll_steps])
        
        # Get target policies and values for initial state and next unroll_steps states
        batch_target_policy.append(traj['policies'][step_idx:step_idx+unroll_steps+1])
        
        # For values, we use actual returns (sum of future rewards) as target
        # This is a simplification - MuZero would use bootstrapped n-step returns
        values = []
        for i in range(step_idx, min(step_idx+unroll_steps+1, traj['steps'])):
            future_return = sum(traj['rewards'][i:])
            values.append(future_return)
        # Pad if necessary
        while len(values) < unroll_steps + 1:
            values.append(0.0)
        batch_target_value.append(values)
        
        # Get target rewards
        rewards = traj['rewards'][step_idx:step_idx+unroll_steps]
        # Pad if necessary
        while len(rewards) < unroll_steps:
            rewards.append(0.0)
        batch_target_reward.append(rewards)
    
    # Convert to tensors and proper format
    batch = {
        "observations": torch.tensor(np.array(batch_observations), dtype=torch.float32),
        "actions": torch.tensor(np.array(batch_actions), dtype=torch.long),
        "target_policy": torch.tensor(np.array(batch_target_policy), dtype=torch.float32),
        "target_value": torch.tensor(np.array(batch_target_value), dtype=torch.float32),
        "target_reward": torch.tensor(np.array(batch_target_reward), dtype=torch.float32)
    }
    
    return batch

def evaluate_network(net_manager, game_manager, num_games=10):
    """Evaluate the network by playing games and measuring average score."""
    total_rewards = []
    
    for _ in range(num_games):
        state = game_manager.generate_initial_state()
        done = False
        total_reward = 0
        
        while not done:
            # Convert state to tensor
            observation_tensor = game_manager.state_to_tensor(state)
            
            # Get network's prediction
            with torch.no_grad():
                network_output = net_manager.initial_inference(observation_tensor)
            
            # Get best action according to policy
            policy = torch.softmax(network_output.policy_logits, dim=1).squeeze(0).numpy()
            legal_actions = game_manager.get_legal_actions(state)
            
            # Mask illegal actions
            legal_policy = np.zeros_like(policy)
            for a in legal_actions:
                legal_policy[a] = policy[a]
                
            # If all legal actions have 0 probability, choose randomly
            if np.sum(legal_policy) == 0:
                action = random.choice(legal_actions)
            else:
                action = np.argmax(legal_policy)
            
            # Take step in the game
            next_state, reward, done = game_manager.get_next_state_and_reward(state, action)
            total_reward += reward
            state = next_state
        
        total_rewards.append(total_reward)
    
    return np.mean(total_rewards)

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Initialize game state manager and neural network
    game_manager = CatchGameStateManager(grid_width=10, grid_height=10)
    net_manager = NeuralNetManager(config)
    
    # Training parameters
    num_iterations = 20    # Number of training iterations
    games_per_iter = 10     # Number of self-play games per iteration
    training_steps = 50     # Number of training steps per iteration
    batch_size = 32         # Batch size for training
    unroll_steps = 5        # Number of steps to unroll for training
    
    # Storage for training data and metrics
    all_trajectories = deque(maxlen=100)  # Store the last 100 games
    training_losses = []
    evaluation_scores = []
    
    print("Starting MuZero training loop...")
    
    # Initial evaluation
    initial_score = evaluate_network(net_manager, game_manager)
    print(f"Initial evaluation score: {initial_score:.2f}")
    evaluation_scores.append(initial_score)
    
    # Training loop
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        start_time = time.time()
        
        # 1. Collect self-play data
        print("Collecting self-play data...")
        new_trajectories = collect_self_play_data(
            net_manager, game_manager, num_games=games_per_iter
        )
        all_trajectories.extend(new_trajectories)
        
        # 2. Train the network
        print("Training network...")
        iteration_losses = []
        
        for step in range(training_steps):
            # Prepare a batch
            batch = prepare_batch(list(all_trajectories), batch_size, unroll_steps)
            if batch is None:
                print("Not enough data for training yet")
                break
                
            # Train on this batch
            loss = net_manager.train_step(batch)
            iteration_losses.append(loss)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}/{training_steps}, Loss: {loss:.4f}")
        
        if iteration_losses:
            avg_loss = sum(iteration_losses) / len(iteration_losses)
            training_losses.append(avg_loss)
            print(f"Average loss this iteration: {avg_loss:.4f}")
        
        # 3. Evaluate the network
        print("Evaluating network...")
        eval_score = evaluate_network(net_manager, game_manager)
        evaluation_scores.append(eval_score)
        print(f"Evaluation score: {eval_score:.2f}")
        
        # Report iteration time
        iteration_time = time.time() - start_time
        print(f"Iteration completed in {iteration_time:.2f} seconds")
        
        # Optional: Save the model periodically
        if (iteration + 1) % 5 == 0:
            save_path = f"muzero_model_iter{iteration+1}.pt"
            net_manager.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Final evaluation
    final_score = evaluate_network(net_manager, game_manager, num_games=20)
    print(f"\nFinal evaluation score (20 games): {final_score:.2f}")
    print(f"Initial evaluation score: {evaluation_scores[0]:.2f}")
    
    # Save final model
    net_manager.save("muzero_model_final.pt")
    print("Final model saved to muzero_model_final.pt")
    
    # Print training statistics
    print("\nTraining summary:")
    print(f"Starting performance: {evaluation_scores[0]:.2f}")
    print(f"Final performance: {evaluation_scores[-1]:.2f}")
    print(f"Improvement: {evaluation_scores[-1] - evaluation_scores[0]:.2f}")
    
    # If matplotlib is available, plot the learning curve
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(training_losses)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(evaluation_scores)
        plt.title('Evaluation Score')
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig('muzero_training_results.png')
        print("Training plots saved to muzero_training_results.png")
    except ImportError:
        print("Matplotlib not available, skipping plots")

if __name__ == "__main__":
    main() 