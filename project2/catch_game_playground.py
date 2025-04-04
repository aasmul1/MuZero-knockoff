from project2.games.catch_game_state_manager import CatchGameStateManager

gs_manager = CatchGameStateManager()
gs_initial_state = gs_manager.generate_initial_state()
gs_available_actions = gs_manager.get_legal_actions(gs_initial_state)
available_actions = gs_manager.get_legal_actions(gs_initial_state)

state = gs_initial_state

done = False

while not done:
    print(state)
    temp_game = gs_manager._create_temp_game(state)

    print(
        f"Fruit height: {temp_game.height - temp_game.fruit_pos[0] - 1}, distance {abs(temp_game.paddle_pos + temp_game.paddle_size // 2 - temp_game.fruit_pos[1])}")
    action = int(input(f"Which action would you like to take? {available_actions} "))
    state, reward, done = gs_manager.get_next_state_and_reward(state, action)
    available_actions = gs_manager.get_legal_actions(state)
    print(f"New state. Reward {reward} from action {action}")
    print()
