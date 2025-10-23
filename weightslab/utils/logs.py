import logging


# Define the log format to include the level, module name, and function name
FORMAT = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'


def setup_logging(level):
    """
    Configures the logging system with the specified severity level.

    kwargs:
        level (int): The minimum level to process
        (e.g., logging.DEBUG, logging.INFO).
    """
    # Reset logger handlers to ensure previous configurations don't interfere
    logging.getLogger().handlers = []

    # Basic logger configuration
    logging.basicConfig(level=level, format=FORMAT)


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
