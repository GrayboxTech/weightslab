import logging
import sys
import tempfile
import atexit
import os
import shutil
from datetime import datetime


# Define the log format to include the level, module name, and function name
FORMAT = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'

# Global variables to track the log file path and handler
_LOG_FILE_PATH = None
_TMP_DIR_PATH = None
_FILE_HANDLER = None


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
    global _TMP_DIR_PATH, _LOG_FILE_PATH, _FILE_HANDLER

    # Reset logger handlers to ensure previous configurations don't interfere
    logging.getLogger().handlers = []

    # Best-effort: reconfigure stdio to UTF-8 (Windows-safe)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Create formatters
    formatter = logging.Formatter(FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(formatter)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    root_logger.addHandler(console_handler)

    # File handler - write to temp directory
    if log_to_file:
        # Create temp directory for logs if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        log_dir = os.path.join(temp_dir, 'weightslab_logs')
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _TMP_DIR_PATH = log_dir
        _LOG_FILE_PATH = os.path.join(log_dir, f'weightslab_{timestamp}.log')

        _FILE_HANDLER = logging.FileHandler(_LOG_FILE_PATH, mode='w', encoding='utf-8')
        _FILE_HANDLER.setLevel(logging.DEBUG)  # Always log DEBUG+ to file
        _FILE_HANDLER.setFormatter(formatter)
        root_logger.addHandler(_FILE_HANDLER)

        # Register exit handler to print log location
        atexit.register(_print_log_location)

        # Log the initialization
        logging.info(f"WeightsLab logging initialized - Log file: {_LOG_FILE_PATH}")


def set_log_directory(new_log_dir):
    """
    Updates the log file location to a new directory.
    Moves the existing log file from temp location to the new directory.
    
    This is automatically called when root_log_dir is resolved in training scripts.
    Can also be called manually if you want to relocate logs.
    
    Args:
        new_log_dir (str): The new directory where logs should be saved.
    
    Example:
        >>> import weightslab as wl
        >>> # Logging starts in temp directory automatically
        >>> # Later, when you define your experiment directory:
        >>> wl.set_log_directory("./my_experiment/logs")
        >>> # Log file is moved from temp to ./my_experiment/logs/
    
    Note:
        - The log file keeps its original timestamped filename
        - All subsequent logs are written to the new location
        - The old temp directory log is moved (not copied)
    """
    global _TMP_DIR_PATH, _LOG_FILE_PATH, _FILE_HANDLER
    
    if not _LOG_FILE_PATH or not _FILE_HANDLER:
        logging.warning("No log file to relocate. Call setup_logging() first.")
        return
    
    # Create new log directory
    os.makedirs(new_log_dir, exist_ok=True)
    
    # Generate new log file path with same filename
    old_filename = os.path.basename(_LOG_FILE_PATH)
    new_log_path = os.path.join(new_log_dir, old_filename)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Flush and close current file handler
    _FILE_HANDLER.flush()
    _FILE_HANDLER.close()
    root_logger.removeHandler(_FILE_HANDLER)
    
    # Move the log file to new location
    try:
        if os.path.exists(_LOG_FILE_PATH):
            shutil.move(_LOG_FILE_PATH, new_log_path)
            logging.info(f"Log file moved from {_LOG_FILE_PATH} to {new_log_path}")
    except Exception as e:
        logging.warning(f"Could not move log file: {e}. Creating new log file at {new_log_path}")
    
    # Update global path
    _LOG_FILE_PATH = new_log_path
    _TMP_DIR_PATH = new_log_dir
    
    # Create new file handler at new location
    formatter = logging.Formatter(FORMAT)
    _FILE_HANDLER = logging.FileHandler(_LOG_FILE_PATH, mode='a', encoding='utf-8')
    _FILE_HANDLER.setLevel(logging.DEBUG)
    _FILE_HANDLER.setFormatter(formatter)
    root_logger.addHandler(_FILE_HANDLER)
    
    logging.info(f"Log directory updated to: {new_log_dir}")
    logging.info(f"Log file: {_LOG_FILE_PATH}")


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
    # Test 1: Setup logging
    setup_logging('DEBUG')
    print('This is a default INFO message')

    # Test 2: Log message at DEBUG level
    print('This message is DEBUG-only', 'and uses sep', sep='|', level='debug')

    # Test 3: Log message at WARNING level
    print('Warning: Something unusual happened.', level='WARNING')

    # Test 4: Relocate log directory
    import tempfile
    new_log_dir = os.path.join(tempfile.gettempdir(), 'weightslab_test_logs')
    print(f'Relocating logs to: {new_log_dir}')
    set_log_directory(new_log_dir)
    
    # Test 5: Log after relocation
    print('This is a message after log relocation.', 'All good.')
    print(f'New log file location: {_LOG_FILE_PATH}')
