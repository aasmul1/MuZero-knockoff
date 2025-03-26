from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.mcts_new import MCTSNew
from project2.models.perfect_model import PerfectModel

gs_manager = CatchGameStateManager()
gs_initial_state = gs_manager.generate_initial_state()
gs_available_actions = gs_manager.get_legal_actions(gs_initial_state)

model = PerfectModel(gs_manager)
mcts = MCTSNew(model, gs_available_actions)

initial_node = model.represent_state(gs_initial_state)
mcts.run_simulations_and_select_action(initial_node)
