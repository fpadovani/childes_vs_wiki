# logger_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(log_file='training.log', max_lines=2000, max_bytes=5*1024*1024):
    """
    Setup a logger that rotates logs after a specific size.
    
    Args:
        log_file (str): The name of the log file.
        max_lines (int): Maximum number of lines per log file (not used, added for future flexibility).
        max_bytes (int): Maximum size in bytes before the log file is rotated.
    
    Returns:
        logging.Logger: Configured logger.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler with rotation
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:  # Avoid adding handlers multiple times
        logger.addHandler(file_handler)


    # Configure basic logging for all processes (console output)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    return logger


