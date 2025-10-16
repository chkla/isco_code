"""
Main pipeline runner for the ISCO classification system.
"""

import os
import logging
# from typing import Dict, Any, Optional, List
# import pandas as pd

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from utils.cache import ResponseCache
from utils.helpers import load_dataframe, save_dataframe, ensure_directory
from data.preprocessing import map_binary_columns, rename_column, clean_predictions
from models.spell_checker import GermanProfessionSpellChecker
from models.vectorstore import create_or_load_vector_store
from pipeline.processing import process_batch
from config import JOB_PROMPT_TEMPLATE, NON_OCCUPATION_TERMS, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


def run_pipeline(
    labeled_file_path: str,
    # additional_labeled_file_path: str,
    unlabeled_file_path: str,
    output_file_path: str,
    run_spellcheck: bool = True,
    force_recreate_vectorstore: bool = False,
    max_workers: int = 5,
    llm_model: str = DEFAULT_LLM_MODEL,
    cache_file: str = 'response_cache.pkl',
    intermediate_output_dir: str = 'intermediate'
) -> bool:
    """
    Run the complete RAG pipeline for occupation classification.

    Args:
        labeled_file_path: Path to main ISCO file
        additional_labeled_file_path: Path to additional labeled data
        unlabeled_file_path: Path to unlabeled data
        output_file_path: Path to save results
        run_spellcheck: Whether to run spell check
        force_recreate_vectorstore: Whether to force recreation of vector store
        max_workers: Number of parallel workers
        llm_model: LLM model to use
        cache_file: Path to cache file
        intermediate_output_dir: Directory to save intermediate files

    Returns:
        True if pipeline ran successfully
    """
    try:
        logger.info("Starting RAG pipeline for occupation classification")

        # Ensure intermediate directory exists
        if not ensure_directory(intermediate_output_dir):
            return False

        # Initialize response cache
        response_cache = ResponseCache(cache_file=cache_file)

        # Step 1: Load unlabeled data
        logger.info(f"Loading unlabeled data from {unlabeled_file_path}")
        df = load_dataframe(unlabeled_file_path, ",", "utf-8")

        # select 10 rows for testing
        # df = df.sample(10, random_state=42)  # :TODO: remove this line for production

        if df is None:
            return False

        # Step 2: Map binary columns
        binary_mapped_path = os.path.join(intermediate_output_dir, 'binary_mapped.csv')
        logger.info("Mapping binary columns")
        df_mapped = map_binary_columns(df)
        save_dataframe(df_mapped, binary_mapped_path)

        # Step 3: Run spell check if enabled
        if run_spellcheck:
            logger.info("Running spell check")
            spell_checker = GermanProfessionSpellChecker(
                model_name=llm_model,
                use_education_context=False,
                cache=response_cache
            )
            df_corrected = spell_checker.process_dataframe(
                df_mapped,
                max_workers=max_workers
            )
            corrected_path = os.path.join(intermediate_output_dir, 'corrected.csv')
            save_dataframe(df_corrected, corrected_path)
        else:
            logger.info("Skipping spell check")
            df_corrected = df_mapped

        # Step 4: Rename corrected column for consistency
        if 'corrected_occupation' in df_corrected.columns:
            logger.info("Renaming corrected_Beschreibung to Beschreibung")
            df_corrected = rename_column(df_corrected, 'corrected_occupation', 'occupation')

        # Step 5: Create or load vector store
        logger.info("Creating or loading vector store")
        vectorstore = create_or_load_vector_store(
            labeled_file_path,
            # additional_labeled_file_path,
            force_recreate=force_recreate_vectorstore
        )

        # Step 6: Set up LLM chain
        logger.info(f"Setting up LLM chain with model {llm_model}")
        llm = ChatOpenAI(model=llm_model)
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=JOB_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
        )

        # Step 7: Process unlabeled data
        logger.info("Processing data with RAG")
        results_df = process_batch(
            df_corrected,
            vectorstore,
            chain,
            response_cache,
            max_workers=max_workers
        )

        # Step 8: Save intermediate results
        raw_output_path = os.path.join(intermediate_output_dir, 'raw_predictions.csv')
        logger.info(f"Saving intermediate results to {raw_output_path}")
        save_dataframe(results_df, raw_output_path)

        # Step 9: Clean predictions
        logger.info("Cleaning predictions")
        cleaned_df = clean_predictions(results_df, list(NON_OCCUPATION_TERMS))

        # Step 10: Save final results
        logger.info(f"Saving final results to {output_file_path}")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not ensure_directory(output_dir):
            return False

        save_dataframe(cleaned_df, output_file_path)

        logger.info(f"Pipeline complete. Final results saved to {output_file_path}")
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False
