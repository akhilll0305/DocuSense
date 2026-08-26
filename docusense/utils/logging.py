"""
Logging configuration for DocuSense.

Provides structured logging with rotation and filtering.
"""

import sys
from loguru import logger

from docusense.config.settings import settings


def setup_logging():
    """
    Configure loguru logger with file rotation and formatting.
    """
    # Remove default logger
    logger.remove()
    
    # Console output with color coding
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File output with rotation
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        enqueue=True,  # Thread-safe
    )
    
    logger.info(f"Logging initialized - Level: {settings.log_level}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Log file: {settings.log_file}")
    
    return logger


# Initialize logging on import
setup_logging()


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Export logger for convenience
__all__ = ["logger", "get_logger", "setup_logging"]
