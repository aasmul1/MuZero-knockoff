import logging

LATENT_DIM = 12
HIDDEN_SIZE = 3
ACTION_SPACE = 3
OBSERVATION_DIM = 75
LEARNING_RATE = 0.001

logging_config = {
    "log_file": "log.log",
    "log_level": logging.DEBUG,
    "plot_backups": False
}
