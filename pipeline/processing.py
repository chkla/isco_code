import logging
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from tenacity import retry, wait_exponential

from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

from utils.cache import ResponseCache
from data.parsing import parse_response, extract_predictions_to_dataframe_row
from models.vectorstore import get_similar_occupations, format_similar_occupations_as_context

logger = logging.getLogger(__name__)


@retry(wait=wait_exponential(multiplier=1000, max=10000))
def process_single_row(
    row: pd.Series,
    vectorstore: FAISS,
    chain: LLMChain,
    response_cache: ResponseCache,
    k: int = 5
) -> pd.Series:
    """
    Process a single occupation description row using RAG.

    Args:
        row: DataFrame row with occupation data
        vectorstore: FAISS vector store
        chain: LangChain LLM chain
        response_cache: Cache for LLM responses
        k: Number of similar documents to retrieve

    Returns:
        Updated row with predictions
    """

    try:
        # Get similar occupations from vector store
        similar_occupations = get_similar_occupations(row['occupation'], vectorstore, k=k)
        print(f"Similar occupations for '{row['occupation']}': {similar_occupations}")

        # Format as context for LLM
        context = format_similar_occupations_as_context(similar_occupations)

        # Run LLM chain to get predictions
        chain_response = chain.run(context=context, question=row['occupation'])
        predictions = parse_response(chain_response)

        # Extract predictions to row format
        prediction_data = extract_predictions_to_dataframe_row(predictions)

        # Add predictions to row
        for key, value in prediction_data.items():
            row[key] = value

        return row

    except Exception as e:
        logger.error(f"Error processing row with description '{row['occupation']}': {e}")
        # Add empty predictions to maintain DataFrame structure
        for i in range(1, 4):
            row[f'predicted_code_{i}'] = ''
            row[f'predicted_berufsbezeichnung_{i}'] = ''
            row[f'confidence_{i}'] = ''
            row[f'reasoning_{i}'] = ''
        return row


def process_batch(
    df: pd.DataFrame,
    vectorstore: FAISS,
    chain: LLMChain,
    response_cache: ResponseCache,
    max_workers: int = 5,
    k: int = 5,
    batch_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Process a batch of occupation descriptions using RAG.

    Args:
        df: DataFrame with occupation descriptions
        vectorstore: FAISS vector store
        chain: LangChain LLM chain
        response_cache: Cache for LLM responses
        max_workers: Number of parallel workers
        k: Number of similar documents to retrieve
        batch_size: Optional batch size (None processes all)

    Returns:
        DataFrame with predictions
    """
    # Apply batch size if specified
    if batch_size is not None and batch_size < len(df):
        logger.info(f"Processing batch of {batch_size} rows")
        df_batch = df.head(batch_size)
    else:
        df_batch = df

    results = []

    logger.info(f"Processing {len(df_batch)} rows with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(
            process_single_row,
            vectorstore=vectorstore,
            chain=chain,
            response_cache=response_cache,
            k=k
        )
        futures = [executor.submit(process_func, row) for _, row in df_batch.iterrows()]

        for future in tqdm(futures, total=len(futures), desc="Processing rows"):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue

    # Get cache statistics
    cache_stats = response_cache.get_stats()

    # Log processing summary
    logger.info("\nProcessing Summary:")
    logger.info(f"Total entries processed: {len(results)}")
    logger.info("Cache Statistics:")
    logger.info(f"Total requests: {cache_stats['total_requests']}")
    logger.info(f"Cache hits: {cache_stats['cache_hits']}")
    logger.info(f"Cache misses: {cache_stats['cache_misses']}")
    logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2f}%")
    logger.info(f"API calls saved: {cache_stats['cache_hits']}")

    return pd.DataFrame(results)
