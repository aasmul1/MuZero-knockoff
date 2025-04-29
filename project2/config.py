import logging

# MuZeroNetwork parameteres
LATENT_DIM = 12
HIDDEN_SIZE = 3
ACTION_SPACE = 3
OBSERVATION_DIM = 75
LEARNING_RATE = 0.001

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 10000  
BATCH_SIZE = 64  
NUM_UNROLL_STEPS = 5  
TD_STEPS = 10  
REPLAY_DISCOUNT = 0.997  

# Catch game parameteres
CATCH_GRID_WIDTH = 5
CATCH_GRID_HEIGHT = 5
PADDLE_SIZE = 2 

# MCTS parameteres
MCTS_C1 = 1.25
MCTS_C2 = 19.562
SIMULATIONS = 10
STEPS = 5
MCTS_DISCOUNT = 0.997 

DISCOUNT = 0.997

# Train muZero config
NUM_ITERATIONS = 10
GAMES_PER_ITERATION = 4
TRAINING_STEPS_PER_ITERATION = 10
CHECKPOINT_FREQUENCY = 10


logging_config = {
    "log_file": "log.log",
    "log_level": logging.DEBUG,
    "plot_backups": False
}