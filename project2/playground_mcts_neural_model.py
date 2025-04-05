import torch

from project2 import config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.mcts import MCTS
from project2.neural_net import NetworkOutput
from project2.neural_net_manager import NeuralNetManager
from project2.node import Node

done = False

gsm = CatchGameStateManager()
nnm = NeuralNetManager(config)
mcts = MCTS(nnm.model)

state = gsm.generate_initial_state()
observation_tensor = gsm.state_to_tensor(state)

print(f"Initial state shape: {state.shape}")
print(f"Observation tensor shape: {observation_tensor.shape}")
print(f"Flattened observation size: {observation_tensor.flatten().shape[0]}")
print(f"Config OBSERVATION_DIM: {config.OBSERVATION_DIM}")
print(f"Config ACTION_SPACE: {config.ACTION_SPACE}")

while not done:
    network_output: NetworkOutput = nnm.initial_inference(observation_tensor)

    print("\nNeural Network Outputs:")
    print(f"Value: {network_output.value.item()}")
    print(f"Policy logits shape: {network_output.policy_logits.shape}")
    print(f"Policy logits {network_output.policy_logits}")

    legal_actions = gsm.get_legal_actions(state)
    print(f"\nLegal actions: {legal_actions}")

    action = mcts.run_simulations_and_select_action(Node(network_output.hidden_state), legal_actions)

    action_tensor = torch.tensor([action.number], dtype=torch.long)

    next_output: NetworkOutput = nnm.recurrent_inference(network_output.hidden_state, action_tensor)

    print("\nAfter action, predicted by network:")
    print(f"Next state value: {next_output.value.item()}")
    print(f"Predicted reward: {next_output.reward.item()}")

    state, reward, done = gsm.get_next_state_and_reward(state, action)
    observation_tensor = gsm.state_to_tensor(state)

    print(f"\nActual next step results:")
    print(f"Reward: {reward}")
    print(f"Game done: {done}")
