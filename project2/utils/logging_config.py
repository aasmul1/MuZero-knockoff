import logging


def configure_logging(log_file="log.log", log_level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logging.info("Logging configured.")
