import logging

LATENT_DIM = 12
HIDDEN_SIZE = 3
ACTION_SPACE = 3
OBSERVATION_DIM = 75
LEARNING_RATE = 0.001

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 10000  # Max number of games to store
BATCH_SIZE = 128  # Number of positions to sample for each update
NUM_UNROLL_STEPS = 5  # Number of steps to unroll during training
TD_STEPS = 10  # Number of steps to bootstrap the return
DISCOUNT = 0.997  # Discount factor for rewards

logging_config = {
    "log_file": "log.log",
    "log_level": logging.DEBUG,
    "plot_backups": False
}