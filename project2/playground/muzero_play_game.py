import logging
import os

from project2.core import config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.core.mcts import MCTS
from project2.models.neural_net_manager import NeuralNetManager
from project2.core.node import Node
from project2.utils.configure_logging import configure_logging
from project2.utils.visualize import visualize_played_game


def main():
    configure_logging(log_file=config.logging_config["log_file"], log_level=config.logging_config["log_level"])
    logger = logging.getLogger("MuZero")

    nnm = NeuralNetManager(config)
    if not os.path.exists(f"models/{config.MODEL_TO_LOAD}"):
        logger.error(f"Model to load does not exist! models/{config.MODEL_TO_LOAD}")
        return

    nnm.load(config.MODEL_TO_LOAD)

    gsm = CatchGameStateManager(config)
    mcts = MCTS(nnm.model, config)

    state = gsm.generate_initial_state()
    observation_tensor = gsm.state_to_tensor(state)
    done = False
    states = [state]

    # Play game
    while not done:
        network_output = nnm.initial_inference(observation_tensor)
        root_node = Node(network_output.hidden_state)
        legal_actions = gsm.get_legal_actions(state)
        action = mcts.run_simulations_and_select_action(root_node, legal_actions)

        next_state, reward, done = gsm.get_next_state_and_reward(state, action)
        state = next_state
        observation_tensor = gsm.state_to_tensor(state)
        states.append(state)

        logger.info(f"New state. Reward {reward}")

    logger.info(f"Done!")

    visualize_played_game(states, plot_file_name=config.FILE_NAME_PLAYED_GAME_PLOT,
                          gif_file_name=config.FILE_NAME_PLAYED_GAME_GIF)


if __name__ == '__main__':
    main()
