import logging

from project2.config import logging_config
from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.logging_config import configure_logging
from project2.mcts import MCTS
from project2.models.perfect_model import PerfectModel

try:
    configure_logging(log_file=logging_config["log_file"], log_level=logging_config["log_level"])
except Exception:
    configure_logging()

logger = logging.getLogger(__name__)

gs_manager = CatchGameStateManager()
gs_initial_state = gs_manager.generate_initial_state()
gs_available_actions = gs_manager.get_legal_actions(gs_initial_state)

model = PerfectModel(gs_manager)
mcts = MCTS(model, gs_available_actions, simulations=800, discount=1)

initial_state = model.represent_state(gs_initial_state)
available_actions = gs_manager.get_legal_actions(initial_state)
state = initial_state
done = False
total_reward = 0

print("Begin")
print(state)
print()

while not done:
    node = mcts.create_node(state)
    action = mcts.run_simulations_and_select_action(node, available_actions)
    state, reward, done = gs_manager.get_next_state_and_reward(state, action)
    available_actions = gs_manager.get_legal_actions(state)
    print(f"New state. Reward {reward} from action {action}\n", state)
    print()
    total_reward += reward

print(f"Total reward: {total_reward}")
