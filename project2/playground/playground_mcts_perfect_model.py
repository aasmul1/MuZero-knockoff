import logging
from copy import deepcopy
from typing import List

from project2.core.config import logging_config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.core.mcts_perfect_model import MCTSPerfectModel
from project2.models.perfect_model import PerfectModel
from project2.core.node import Node
from project2.utils.configure_logging import configure_logging
from project2.utils.visualize import visualize_search_tree

try:
    configure_logging(log_file=logging_config["log_file"], log_level=logging_config["log_level"])
except Exception:
    configure_logging()

logger = logging.getLogger(__name__)

gs_manager = CatchGameStateManager()
gs_initial_state = gs_manager.generate_initial_state()
gs_available_actions = gs_manager.get_legal_actions(gs_initial_state)

model = PerfectModel(gs_manager)
mcts = MCTSPerfectModel(model, gs_available_actions, simulations=10, discount=1, steps=100, c_1=2, c_2=20000)

initial_state = model.represent_state(gs_initial_state)
available_actions = gs_manager.get_legal_actions(initial_state)
state = initial_state

done = False
total_reward = 0
runs = 0
states = list(state)

while not done:
    node = mcts.create_node(state)
    action = mcts.run_simulations_and_select_action(node, available_actions)
    state, reward, done = gs_manager.get_next_state_and_reward(state, action)
    available_actions = gs_manager.get_legal_actions(state)
    print(f"New state. Reward {reward} from action {action}")
    states.append(state)
    total_reward += reward

    runs += 1
    if runs > 100:
        done = True

print(f"Total reward: {total_reward}")

root_nodes: List[Node] = deepcopy(mcts.root_nodes)
root_nodes.append(mcts.create_node(state))

visualize_search_tree(search_tree=mcts.search_tree, root_nodes=root_nodes, reward_table=mcts.reward_table,
                      policy_table=mcts.policy_table, visit_count_table=mcts.visit_count_table,
                      mean_value_table=mcts.mean_value_table,
                      state_transition_table=mcts.state_transition_table, ucb_scores=mcts.ucb_scores)
