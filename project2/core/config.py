import logging

# MuZeroNetwork parameteres
LATENT_DIM = 64
HIDDEN_SIZE = 128
ACTION_SPACE = 3
OBSERVATION_DIM = 147
LEARNING_RATE = 0.001

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 128
NUM_UNROLL_STEPS = 3
TD_STEPS = 10
REPLAY_DISCOUNT = 1

# Catch game parameteres
CATCH_GRID_WIDTH = 7
CATCH_GRID_HEIGHT = 7
PADDLE_SIZE = 1
MAX_TOTAL_REWARDS = 10  # Maximum number of total rewards before ending game, i.e., fruits catched

# MCTS parameteres
MCTS_C1 = 1.4
MCTS_C2 = 2000
SIMULATIONS = 100
STEPS = 100
MCTS_DISCOUNT = 1

DISCOUNT = 1

# Train MuZero config
NUM_ITERATIONS = 500
GAMES_PER_ITERATION = 20
TRAINING_STEPS_PER_ITERATION = 30  # Have tried 100
CHECKPOINT_FREQUENCY = 10

logging_config = {
    "log_file": "muzero_training.log",
    "log_level": logging.INFO,
    "plot_backups": False
}

# MuZero play catch game
MODEL_TO_LOAD = "muzero_1.pt"
FILE_NAME_PLAYED_GAME_PLOT = "played_game.png"
FILE_NAME_PLAYED_GAME_GIF = "played_game.gif"
NUM_GAMES_TO_PLAY = 10
