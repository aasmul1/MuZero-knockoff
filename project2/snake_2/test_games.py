import time
from snake import SnakeGame
#from catch import CatchGame
from pynput import keyboard

def manual_test_snake():
    """test snake manuelt"""
    game = SnakeGame(grid_size=10, grow_on_food=True)
    game_over = False
    1
    # initialize
    state = game.reset()
    score = 0
    
    def on_press(key):
        nonlocal game_over, state, score
        
        if game_over:
            return False
        
        try:
            # map taster til actions
            action = None
            if key == keyboard.Key.left:
                action = 0  # VENSTRE
            elif key == keyboard.Key.up:
                action = 1  # OPP
            elif key == keyboard.Key.right:
                action = 2  # HØYRE
            elif key == keyboard.Key.down:
                action = 3  # NED
            elif key == keyboard.Key.esc:
                game_over = True
                return False
            
            if action is not None:
                # utfør action
                state, reward, game_over = game.step(action)
                if reward > 0:
                    score += reward
                
                # tøm skjerm og render
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                game.render()
                
                if game_over:
                    print("Game Over! Final score:", score)
                    return False
                
        except Exception as e:
            print(f"Error: {e}")
    
    # initial rendering
    game.render()
    print("Bruk piltastene for å styre slangen Trykk ESC for å avslutte")
    
    # start lytting etter tast
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# def manual_test_catch():
#     """test catch-spillet manuelt"""
#     game = CatchGame(grid_width=10, grid_height=10)
#     game_over = False
    
#     # initialiser state
#     state = game.reset()
#     score = 0
    
#     def on_press(key):
#         nonlocal game_over, state, score
        
#         if game_over:
#             return False
        
#         try:
#             # mapp taster til handlinger
#             action = None
#             if key == keyboard.Key.left:
#                 action = 0  # VENSTRE
#             elif key == keyboard.Key.space:
#                 action = 1  # STÅ STILLE
#             elif key == keyboard.Key.right:
#                 action = 2  # HØYRE
#             elif key == keyboard.Key.esc:
#                 game_over = True
#                 return False
            
#             if action is not None:
#                 # utfør action
#                 state, reward, game_over = game.step(action)
#                 if reward > 0:
#                     score += reward
                
#                 # tøm skjerm og render
#                 import os
#                 os.system('cls' if os.name == 'nt' else 'clear')
#                 game.render()
                
#                 if game_over:
#                     print("Game Over! Final score:", score)
#                     return False
                
#         except Exception as e:
#             print(f"Error: {e}")
    
#     # initial rendering
#     game.render()
#     print("Bruk VENSTRE/HØYRE piltaster for å flytte racket Trykk MELLOMROM for å stå stille Trykk ESC for å avslutte")
    
#     # start lytting etter tastaturtrykk
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()

if __name__ == "__main__":
    print("Velg spill å teste:")
    print("1. Snake")
    print("2. Catch")
    choice = input("Skriv inn valg (1/2): ")
    
    if choice == "1":
        manual_test_snake()
    # elif choice == "2":
    #     manual_test_catch()
    else:
        print("Ugyldig valg!")