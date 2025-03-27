import argparse
#from rl_manager import RLManager
from snake_game_state_manager import SnakeGameStateManager
from catch_game_state_manager import CatchGameStateManager
from visualizer import GameVisualizer
import test_games

def main():
    parser = argparse.ArgumentParser(description='MuZero for Snake and Catch games')
    parser.add_argument('--game', type=str, default='snake', choices=['snake', 'catch'],
                        help='spill (snake eller catch)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'manual'],
                        help='modus: train, test eller manual play')
    parser.add_argument('--visualize', action='store_true',
                        help='visualiser gameplay under trening/testing')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='antall episoder for trening')
    
    args = parser.parse_args()
    
    # manuell 
    if args.mode == 'manual':
        if args.game == 'snake':
            test_games.manual_test_snake()
        else:
            test_games.manual_test_catch()
        return
    
    # lag game state manager basert på valgt game
    if args.game == 'snake':
        game_state_manager = SnakeGameStateManager(grid_size=5, grow_on_food=True)
    else:
        game_state_manager = CatchGameStateManager(grid_width=5, grid_height=5)
    
    # # lag visualizer
    visualizer = GameVisualizer(game_type=args.game)
    
    # lag RL-manager og train/test
    #rl_manager = RLManager(game_state_manager, visualizer)
    
    # if args.mode == 'train':
    #     rl_manager.train(args.episodes, visualize=args.visualize)
    # else:  # test mode
    #     rl_manager.test(visualize=True)

if __name__ == "__main__":
    main()