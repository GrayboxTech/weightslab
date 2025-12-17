import logging
import tempfile
import atexit
import os
from datetime import datetime


# Define the log format to include the level, module name, and function name
FORMAT = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'

# Global variable to track the log file path
_LOG_FILE_PATH = None


def _print_log_location():
    """Print log file location when Python exits."""
    if _LOG_FILE_PATH and os.path.exists(_LOG_FILE_PATH):
        print(f"\n{'='*60}\nWeightsLab session log saved to:\n{_LOG_FILE_PATH}\n{'='*60}", flush=True)


def setup_logging(level, log_to_file=True):
    """
    Configures the logging system with the specified severity level.
    Automatically writes logs to a temporary directory if log_to_file is True.

    kwargs:
        level (str): The minimum level to process (e.g., 'DEBUG', 'INFO').
        log_to_file (bool): If True, logs are written to a temp file (default: True).
    """
    global _LOG_FILE_PATH

    # Reset logger handlers to ensure previous configurations don't interfere
    logging.getLogger().handlers = []

    # Create formatters
    formatter = logging.Formatter(FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(formatter)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.addHandler(console_handler)

    # File handler - write to temp directory
    if log_to_file:
        # Create temp directory for logs if it doesn't exist
        temp_dir = tempfile.gettempdir()
        log_dir = os.path.join(temp_dir, 'weightslab_logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _LOG_FILE_PATH = os.path.join(log_dir, f'weightslab_{timestamp}.log')

        file_handler = logging.FileHandler(_LOG_FILE_PATH, mode='w')
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG+ to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Register exit handler to print log location
        atexit.register(_print_log_location)

        # Log the initialization
        logging.info(f"WeightsLab logging initialized - Log file: {_LOG_FILE_PATH}")


def print(first_element, *other_elements, sep=' ', **kwargs):
    """
    Overrides the built-in print function to use logging features.

    The output level (DEBUG, INFO, WARNING, etc.) can be controlled
    using the 'level' keyword argument. Defaults to 'INFO'.

    Args:
        first_element: The mandatory first element to log.
        *other_elements: All subsequent positional arguments.
        sep (str): The separator to use between elements (default: ' ').
        **kwargs: Optional keyword arguments, including 'level' to set the
        severity.
    """
    # 0. Setup logging
    level_str = kwargs.pop('level', 'INFO').upper()

    # 1. Combine all positional elements into a single log message string.
    all_elements = (first_element,) + other_elements
    log_message = sep.join(map(str, all_elements))

    # 2. Map the string level to the corresponding logging method.
    if level_str == "DEBUG":
        logging.debug(log_message)
    elif level_str == "INFO":
        logging.info(log_message)
    elif level_str == "WARNING":
        logging.warning(log_message)
    elif level_str == "ERROR":
        logging.error(log_message)
    elif level_str == "CRITICAL":
        logging.critical(log_message)
    else:
        # Default fallback if an unknown level is provided
        logging.info(log_message)


if __name__ == "__main__":
    # Setup prints
    setup_logging('DEBUG')

    print('This is a default INFO message')

    # 2. Log message at DEBUG level
    print('This message is DEBUG-only', 'and uses sep', sep='|', level='debug')

    # 3. Log message at WARNING level
    print('Warning: Something unusual happened.', level='WARNING')

    # 4. Standard logging INFO message (no level specified)
    print('This is a final INFO message.', 'All good.')
