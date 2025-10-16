"""
Logging configuration for the ISCO classification system.
"""

import os
import logging
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, creates timestamped file)

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Generate default log filename with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/isco_classifier_{timestamp}.log"

    # Configure logging
    handlers = [
        logging.StreamHandler(),  # Console handler
        logging.FileHandler(log_file)  # File handler
    ]

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Create and return logger
    logger = logging.getLogger("isco_classifier")
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger
