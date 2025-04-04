import torch

from project2 import config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.neural_net_manager import NeuralNetManager


def main():
    # Initialize game state manager
    game_manager = CatchGameStateManager(grid_width=10, grid_height=10)

    # Initialize neural network manager
    net_manager = NeuralNetManager(config)

    # Get initial state
    state = game_manager.generate_initial_state()

    # Convert state to tensor format expected by neural network
    observation_tensor = game_manager.state_to_tensor(state)

    print(f"Initial state shape: {state.shape}")
    print(f"Observation tensor shape: {observation_tensor.shape}")
    print(f"Flattened observation size: {observation_tensor.flatten().shape[0]}")
    print(f"Config OBSERVATION_DIM: {config.OBSERVATION_DIM}")
    print(f"Config ACTION_SPACE: {config.ACTION_SPACE}")

    # Run initial inference through the neural network
    network_output = net_manager.initial_inference(observation_tensor)

    # Display results
    print("\nNeural Network Outputs:")
    print(f"Value: {network_output.value.item()}")
    print(f"Policy logits shape: {network_output.policy_logits.shape}")
    print(f"Policy logits {network_output.policy_logits}")

    # Get legal actions
    legal_actions = game_manager.get_legal_actions(state)
    print(f"\nLegal actions: {legal_actions}")

    # Choose an action (for this example, just take the first legal action)
    action = legal_actions[0]

    # Convert action to tensor for recurrent inference
    action_tensor = torch.tensor([action], dtype=torch.long)

    # Run recurrent inference
    next_output = net_manager.recurrent_inference(network_output.hidden_state, action_tensor)

    print("\nAfter action, predicted by network:")
    print(f"Next state value: {next_output.value.item()}")
    print(f"Predicted reward: {next_output.reward.item()}")

    # Take the action in the actual game
    next_state, actual_reward, done = game_manager.get_next_state_and_reward(state, action)

    print(f"\nActual next step results:")
    print(f"Reward: {actual_reward}")
    print(f"Game done: {done}")

    print("\nThis demonstrates the connection between the game and neural network!")


if __name__ == "__main__":
    main()
