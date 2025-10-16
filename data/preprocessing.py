"""
Data preprocessing functions for the ISCO classification system.
"""

import os
import csv
import logging
import pandas as pd
from typing import List, Dict  # Any, Optional

logger = logging.getLogger(__name__)


def preprocess_csv(file_path: str, delimiter: str = ';') -> List[Dict[str, str]]:
    """
    Extract and preprocess job data from CSV.

    Args:
        file_path: Path to CSV file
        delimiter: CSV delimiter

    Returns:
        List of dictionaries with job data
    """
    job_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                job = {
                    'isco': row.get('isco', ''),
                    'occupation': row.get('occupation', '')
                }
                job_data.append(job)
        logger.info(f"Preprocessed {len(job_data)} job entries from {file_path}")
        return job_data
    except Exception as e:
        logger.error(f"Error preprocessing CSV {file_path}: {e}")
        raise


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize job description data.

    Args:
        df: DataFrame with job descriptions

    Returns:
        Cleaned DataFrame
    """
    # Check if required columns exist
    if not all(col in df.columns for col in ["occupation", "isco"]):
        raise ValueError("DataFrame must contain columns 'occupation' and 'isco'")

    cleaned_df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    cleaned_df["occupation"] = cleaned_df["occupation"].astype(str).apply(
        lambda x: x.title() if isinstance(x, str) else x
    )  # Convert to title case
    cleaned_df = cleaned_df[["occupation", "isco"]]  # Select relevant columns

    return cleaned_df


def map_binary_columns(df: pd.DataFrame, columns: List[str] = ["training", "unidegree"], mapping: Dict[int, str] = {0: "no", 1: "yes"}) -> pd.DataFrame:
    """
    Map binary 0/1 values to 'no'/'yes' in specified columns.

    Args:
        df: DataFrame with columns to map
        columns: List of column names to map
        mapping: Dictionary mapping values

    Returns:
        DataFrame with mapped columns
    """
    result_df = df.copy()

    for col in columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].map(mapping)
            logger.debug(f"Mapped values in column '{col}'")

    return result_df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """
    Rename column in DataFrame.

    Args:
        df: DataFrame with column to rename
        old_name: Column name to rename
        new_name: New column name

    Returns:
        DataFrame with renamed column
    """
    result_df = df.copy()

    # If new column already exists, drop it
    if new_name in result_df.columns:
        result_df = result_df.drop(columns=[new_name])

    # Rename column
    result_df = result_df.rename(columns={old_name: new_name})
    logger.debug(f"Renamed column '{old_name}' to '{new_name}'")

    return result_df


def clean_predictions(input_data, output_or_terms=None, non_occupation_terms=None):
    """
    Filter out non-occupation entries from predictions.

    This function can be called in three ways:
    1. clean_predictions(df) - Just clean the DataFrame
    2. clean_predictions(df, non_occupation_terms) - Clean with specific terms
    3. clean_predictions(input_file, output_file) - Load, clean and save file

    Args:
        input_data: DataFrame or path to input predictions file
        output_or_terms: Either output file path or list of non-occupation terms
        non_occupation_terms: List of terms indicating non-occupations

    Returns:
        Cleaned DataFrame
    """

    logger = logging.getLogger(__name__)

    # Figure out the argument pattern being used
    output_file = None

    # If the second argument is a list, it's the terms
    if isinstance(output_or_terms, (list, set, tuple)):
        non_occupation_terms = output_or_terms
    # If it's a string, it might be a file path (for file input)
    elif isinstance(output_or_terms, str) and isinstance(input_data, str):
        output_file = output_or_terms

    # Default terms if none provided
    if non_occupation_terms is None:
        try:
            from config import NON_OCCUPATION_TERMS
            non_occupation_terms = list(NON_OCCUPATION_TERMS)
        except ImportError:
            non_occupation_terms = [
                'student', 'studentin', 'studenten', 'studentinnen',
                'schüler', 'schülerin', 'schülerinnen',
                'arbeitslos', 'arbeitslose', 'arbeitssuchend',
                'studiere', 'studierend', 'studierende', 'studierenden',
                'promovierende', 'promovend', 'promovendin', 'doktorand', 'doktorandin',
                'azubi', 'auszubildende', 'auszubildender',
                'rentner', 'rentnerin', 'pensionär', 'pensionärin',
                'elternzeit', 'mutterschutz', 'hausfrau', 'hausmann',
                'arbeitssuchende', 'arbeitsuchend', 'arbeitssuchend',
                'praktikant', 'praktikantin', '-97 möchte ich nicht beantworten',
                'Arbeitslos', 'Student'
            ]

    try:
        # Check if input is a DataFrame or a file path
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            logger.info(f"Processing DataFrame with {len(df)} rows")
        elif isinstance(input_data, str) and os.path.isfile(input_data):
            df = pd.read_csv(input_data)
            logger.info(f"Loaded predictions from {input_data} with {len(df)} rows")
        else:
            error_msg = f"Invalid input: {type(input_data)}. Expected DataFrame or file path."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle each predicted_code column individually
        for i in range(1, 4):
            col_name = f'predicted_code_{i}'
            if col_name not in df.columns:
                continue

            # Convert to string first to ensure consistent handling
            df[col_name] = df[col_name].astype(str)

            # Identify rows with numeric values
            numeric_mask = df[col_name].str.replace('.', '', 1).str.isdigit()

            # Convert numeric values to int
            if numeric_mask.any():
                # First convert to float (to handle potential decimals), then to int
                df.loc[numeric_mask, col_name] = (
                    pd.to_numeric(df.loc[numeric_mask, col_name], errors='coerce')
                    .fillna(0)
                    .astype(int)
                    .astype(str)  # Convert back to string for consistency
                )

        # Create mask for rows to clean (if Beschreibung column exists)
        if 'Beschreibung' in df.columns:
            mask = df['Beschreibung'].astype(str).str.lower().apply(
                lambda x: any(k in x.lower() for k in non_occupation_terms)
            )
            logger.info(f"Found {mask.sum()} rows matching non-occupation terms")
        else:
            mask = pd.Series(False, index=df.index)
            logger.warning("No 'Beschreibung' column found for filtering non-occupations")

        # Define columns to clean
        cols = []
        for i in range(1, 4):
            for prefix in ['predicted_code_', 'predicted_berufsbezeichnung_', 'confidence_', 'reasoning_']:
                col = f'{prefix}{i}'
                if col in df.columns:
                    cols.append(col)

        # Clean the data
        if mask.any() and cols:
            df.loc[mask, cols] = ''
            logger.info(f"Cleaned {mask.sum()} rows in {len(cols)} columns")

        # Save to file if output_file is provided
        if output_file is not None:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved cleaned predictions to {output_file}")

        return df

    except Exception as e:
        logger.error(f"Error cleaning predictions: {e}")
        raise


def is_valid_occupation(text: str, non_occupation_terms: List[str]) -> bool:
    """
    Check if the given text represents a valid occupation.

    Args:
        text: Text to check
        non_occupation_terms: List of terms indicating non-occupations

    Returns:
        bool: False if text contains non-occupation terms or patterns
    """

    # Check for specific non-occupation terms
    words = text.split()
    if any(word in non_occupation_terms for word in words):
        logger.debug(f"Text '{text}' contains non-occupation terms.")
        return False

    return True
