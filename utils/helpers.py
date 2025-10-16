"""
General utility functions for the ISCO classification system.
"""

import os
import logging
import pandas as pd
from typing import Optional, List  # , Dict, Any, Tuple

logger = logging.getLogger(__name__)


def ensure_directory(directory_path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to directory

    Returns:
        True if directory exists or was created successfully
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    return True


def load_dataframe(file_path: str, delimiter: str = ';', encoding: str = 'utf-8') -> Optional[pd.DataFrame]:
    """
    Load a DataFrame from a CSV file with error handling.

    Args:
        file_path: Path to CSV file
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        DataFrame or None if loading failed
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        logger.info(f"Loaded DataFrame from {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to load DataFrame from {file_path}: {e}")
        return None


def save_dataframe(df: pd.DataFrame, file_path: str, delimiter: str = ';', index: bool = False) -> bool:
    """
    Save a DataFrame to a CSV file with error handling.

    Args:
        df: DataFrame to save
        file_path: Path to save CSV file
        delimiter: CSV delimiter
        index: Whether to include index

    Returns:
        True if saved successfully
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not ensure_directory(directory):
            return False

        # Save DataFrame
        df.to_csv(file_path, sep=delimiter, index=index)
        logger.info(f"Saved DataFrame with {len(df)} rows to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {file_path}: {e}")
        return False


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log information about a DataFrame.

    Args:
        df: DataFrame to log info about
        name: Name to use in log message
    """
    try:
        logger.info(f"{name} info:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")

        # Sample data if dataframe is not empty
        if len(df) > 0:
            logger.debug(f"  Sample data:\n{df.head(3)}")
    except Exception as e:
        logger.error(f"Error logging DataFrame info: {e}")


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file

    Returns:
        True if file exists
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    return True


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if DataFrame has all required columns
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    return True
