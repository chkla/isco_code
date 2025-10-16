"""
Response parsing functions for the ISCO classification system.
"""

import re
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

def parse_response(response: str) -> List[Dict[str, str]]:
    """
    Parse LLM response into structured predictions.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        List of dictionaries with structured predictions
    """
    predictions = []
    current_pred = {}

    # Use regex to find all instances of "Code:", "Berufsbezeichnung:", and "Confidence:"
    code_pattern = re.compile(r'Code:\s*([\w\d]+)')
    berufsbezeichnung_pattern = re.compile(r'Berufsbezeichnung:\s*(.+?)\s*(?=,\s*Confidence:|$)')
    confidence_pattern = re.compile(r'Confidence:\s*([\d.]+)')

    try:
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.')):
                if current_pred:
                    predictions.append(current_pred)
                    current_pred = {}

                code_match = code_pattern.search(line)
                berufsbezeichnung_match = berufsbezeichnung_pattern.search(line)
                confidence_match = confidence_pattern.search(line)

                if code_match:
                    current_pred['ISCO'] = code_match.group(1)
                if berufsbezeichnung_match:
                    current_pred['Beschreibung'] = berufsbezeichnung_match.group(1)
                if confidence_match:
                    current_pred['confidence'] = confidence_match.group(1)

            elif line.strip().startswith('Reasoning:'):
                current_pred['reasoning'] = line.split('Reasoning:')[1].strip()

        # Add the last prediction if it exists
        if current_pred:
            predictions.append(current_pred)

        # Validate predictions
        if not predictions:
            logger.warning("No predictions found in response")
            
        for i, pred in enumerate(predictions):
            if 'ISCO' not in pred:
                logger.warning(f"Prediction {i+1} missing ISCO code")
            if 'Beschreibung' not in pred:
                logger.warning(f"Prediction {i+1} missing Beschreibung")
            if 'confidence' not in pred:
                logger.warning(f"Prediction {i+1} missing confidence score")
                
        return predictions
        
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        logger.debug(f"Problematic response: {response}")
        return []

def extract_predictions_to_dataframe_row(predictions: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Extract predictions into a format suitable for a DataFrame row.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dictionary with predictions formatted for DataFrame row
    """
    row_data = {}
    
    # Initialize with empty values for all potential columns
    for i in range(1, 4):
        row_data[f'predicted_code_{i}'] = ''
        row_data[f'predicted_berufsbezeichnung_{i}'] = ''
        row_data[f'confidence_{i}'] = ''
        row_data[f'reasoning_{i}'] = ''
    
    # Add predictions to row data
    for i, pred in enumerate(predictions[:3], 1):  # Limit to first 3 predictions
        row_data[f'predicted_code_{i}'] = pred.get('ISCO', '')
        row_data[f'predicted_berufsbezeichnung_{i}'] = pred.get('Beschreibung', '')
        row_data[f'confidence_{i}'] = pred.get('confidence', '')
        row_data[f'reasoning_{i}'] = pred.get('reasoning', '')
    
    return row_data
