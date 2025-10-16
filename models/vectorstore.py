"""
Vector store operations for the ISCO classification system.
"""

import os
import logging
from typing import List, Dict, Any  # Optional,
# import pandas as pd

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from data.preprocessing import preprocess_csv  # , clean_table
from config import VECTOR_STORE_PATH, DEFAULT_EMBEDDINGS_MODEL

logger = logging.getLogger(__name__)


def create_or_load_vector_store(
    file_path: str,
    force_recreate: bool = False,
    vector_store_path: str = VECTOR_STORE_PATH,
    embeddings_model: str = DEFAULT_EMBEDDINGS_MODEL
) -> FAISS:
    """
    Create or load vector store for occupation embeddings.

    Args:
        file_path: Path to main ISCO file
        additional_labeled_file_path: Path to additional labeled data
        force_recreate: Whether to force recreation of vector store
        vector_store_path: Path to store/load vector store
        embeddings_model: OpenAI embeddings model to use

    Returns:
        FAISS vector store
    """
    if os.path.exists(vector_store_path) and not force_recreate:
        logger.info("Loading existing vector store...")
        try:
            embeddings = OpenAIEmbeddings(model=embeddings_model)
            return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except ValueError as e:
            logger.error(f"Error loading vector store: {e}")
            logger.info("Recreating vector store...")
            force_recreate = True

    # Process when force_recreate is true or vector store doesn't exist
    if force_recreate or not os.path.exists(vector_store_path):
        logger.info("Creating new vector store...")

        # Load and process the main ISCO data
        job_data = preprocess_csv(file_path, delimiter=';')  # :TODO: attention typical error, adjust delimiter!

        # Combine documents from both sources
        documents = [
            Document(
                page_content=job['occupation'],
                metadata={'isco': job['isco'], 'occupation': job['occupation']}
            ) for job in job_data
        ]

        logger.info(f"Creating embeddings for {len(documents)} documents using model {embeddings_model}")
        embeddings = OpenAIEmbeddings(model=embeddings_model)
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(vector_store_path) or '.', exist_ok=True)

        logger.info(f"Saving vector store to {vector_store_path}")
        vectorstore.save_local(vector_store_path)

        return vectorstore


def get_similar_occupations(
    query: str,
    vectorstore: FAISS,
    k: int = 5
) -> List[Dict[str, Any]]:
    """
    Get similar occupations for a query from the vector store.

    Args:
        query: Occupation description to find similar occupations for
        vectorstore: FAISS vector store
        k: Number of similar occupations to return

    Returns:
        List of dictionaries with similar occupations
    """
    try:
        similar_docs = vectorstore.similarity_search_with_score(query, k=k)

        # Format results
        results = []
        for doc, score in similar_docs:
            results.append({
                'isco': doc.metadata['isco'],
                'occupation': doc.metadata['occupation'],
                'similarity_score': score
            })

        return results
    except Exception as e:
        logger.error(f"Error getting similar occupations for '{query}': {e}")
        return []


def format_similar_occupations_as_context(similar_occupations: List[Dict[str, Any]]) -> str:
    """
    Format similar occupations as context string for LLM prompt.

    Args:
        similar_occupations: List of similar occupation dictionaries

    Returns:
        Formatted context string
    """
    context_lines = []

    for i, occ in enumerate(similar_occupations, 1):
        score_percent = (1 - occ.get('similarity_score', 0)) * 100  # Convert to percentage
        context_lines.append(
            f"{i}. isco: {occ['isco']}, occupation: {occ['occupation']} "
            f"(Similarity: {score_percent:.1f}%)"
        )

    return "\n".join(context_lines)
