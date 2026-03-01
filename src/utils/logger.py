"""
Logging Configuration

Configures loguru logger for the application.
"""

from loguru import logger
import sys


def setup_logger(log_level="INFO", log_file=None):
    """
    Configure the logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days"
        )

    logger.info(f"Logger initialized with level={log_level}")
