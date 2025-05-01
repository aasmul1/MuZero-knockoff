import logging

from project2.core.buffer_game import Game
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.core.mcts import MCTS
from project2.models.neural_net import NetworkOutput
from project2.core.node import Node

logger = logging.getLogger("game")


def play_and_record_game(config, neural_net_manager):
    gsm = CatchGameStateManager(config)
    mcts = MCTS(neural_net_manager.model, config)

    game = Game()

    state = gsm.generate_initial_state()
    observation_tensor = gsm.state_to_tensor(state)
    done = False

    while not done:

        game.observations.append(observation_tensor.clone())
        network_output: NetworkOutput = neural_net_manager.initial_inference(observation_tensor)

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

    return game, gsm
