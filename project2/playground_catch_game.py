from project2.games.catch_game_state_manager import CatchGameStateManager
from project2.models.perfect_model import PerfectModel

gsm = CatchGameStateManager()
gs_initial_state = gsm.generate_initial_state()
gs_available_actions = gsm.get_legal_actions(gs_initial_state)
available_actions = gsm.get_legal_actions(gs_initial_state)
model = PerfectModel(gsm)

state = gs_initial_state

done = False

while not done:
    print(state)
    print(model.predict(model.represent_state(state)))
    temp_game = gsm._create_temp_game(state)
    print(
        f"Fruit height: {temp_game.height - temp_game.fruit_pos[0] - 1}, distance {abs(temp_game.paddle_pos + temp_game.paddle_size // 2 - temp_game.fruit_pos[1])}")
    action = int(input(f"Which action would you like to take? {available_actions} "))
    state, reward, done = gsm.get_next_state_and_reward(state, action)
    available_actions = gsm.get_legal_actions(state)
    print(f"New state. Reward {reward} from action {action}")
    print()
