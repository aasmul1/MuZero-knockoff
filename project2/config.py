import logging

# MuZeroNetwork parameteres
LATENT_DIM = 64
HIDDEN_SIZE = 128
ACTION_SPACE = 3
OBSERVATION_DIM = 147
LEARNING_RATE = 0.0005

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 128
NUM_UNROLL_STEPS = 5
TD_STEPS = 10
REPLAY_DISCOUNT = 0.997

# Catch game parameteres
CATCH_GRID_WIDTH = 7
CATCH_GRID_HEIGHT = 7
PADDLE_SIZE = 2

# MCTS parameteres
MCTS_C1 = 1.25
MCTS_C2 = 19.562
SIMULATIONS = 50
STEPS = 10
MCTS_DISCOUNT = 0.997

DISCOUNT = 0.997

# Train muZero config
NUM_ITERATIONS = 50
GAMES_PER_ITERATION = 20
TRAINING_STEPS_PER_ITERATION = 30
CHECKPOINT_FREQUENCY = 10

logging_config = {
    "log_file": "muzero_training.log",
    "log_level": logging.INFO,
    "plot_backups": False
}
