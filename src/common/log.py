import logging
import sys


def get_logger(module_name: str) -> logging.Logger:
    """Get logger for a given module.

    Args:
        module_name: Name of the module (__name__)
    """
    logger = logging.getLogger(module_name)
    date_strftime_format = "%Y-%m-%y %H:%M:%S"
    format = (
        "[%(levelname)s] %(asctime)s | %(message)s | %(name)s-%(funcName)s:%(lineno)d"
    )
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=format,
        datefmt=date_strftime_format,
    )
    return logger
