import logging


def setup_logger():
    """
    Set up a logger named 'ModelLogger' with DEBUG level logging to the console.

    Returns:
    logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("ModelLogger")
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)

    return logger
