import torch

from project2 import config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.mcts import MCTS
from project2.neural_net_manager import NeuralNetManager
from project2.neural_net import NetworkOutput
from project2.node import Node
from project2.replay_buffer import ReplayBuffer
from project2.game import Game


def play_and_record_game():

    gsm = CatchGameStateManager()
    nnm = NeuralNetManager(config)
    mcts = MCTS(nnm.model)
    replay_buffer = ReplayBuffer(config)
    
    game = Game()

    state = gsm.generate_initial_state()
    observation_tensor = gsm.state_to_tensor(state)
    done = False

    while not done:
        
        game.observation.append(observation_tensor.clone())
        network_output: NetworkOutput = nnm.initial_inference(observation_tensor)

        root_node = Node(network_output.hidden_state)
        
        legal_actions = gsm.get_legal_actions(state)

        action = mcts.run_simulations_and_select_action(root_node, legal_actions)
        
        visit_counts = {}
        for a in legal_actions:
            visit_counts[a] = mcts.visit_count_table.get((root_node, a), 0)
            
        game.store_search_statistics(visit_counts, root_node)
        
        game.values.append(network_output.value.item())
        
        next_state, reward, done = gsm.get_next_state_and_reward(state, action)
        
        game.actions.append(action)
        game.rewards.append(reward)
        
        state = next_state
        
        observation_tensor = gsm.state_to_tensor(state)
    
    replay_buffer.save_game(game)
    
    return replay_buffer

# Example usage
if __name__ == "__main__":
    replay_buffer = play_and_record_game()
    # Now we can sample from the buffer for training
    batch = replay_buffer.sample_batch()
    print(f"Sampled batch with {len(batch['observations'])} examples")