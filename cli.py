import os
import argparse
import logging
import sys

from utils.logging_config import setup_logging
from pipeline.runner import run_pipeline
from config import DEFAULT_PATHS, DEFAULT_LLM_MODEL, DEFAULT_MAX_WORKERS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Occupation classification using RAG')

    # Input and output files
    parser.add_argument('--labeled-file',
                        default=DEFAULT_PATHS['labeled_file'],
                        help='Path to main ISCO file')
    parser.add_argument('--unlabeled-file',
                        default=DEFAULT_PATHS['unlabeled_file'],
                        help='Path to unlabeled data')
    parser.add_argument('--output-file',
                        default=DEFAULT_PATHS['output_file'],
                        help='Path to save results')

    # Processing options
    parser.add_argument('--skip-spellcheck',
                        action='store_true',
                        help='Skip spell check')
    parser.add_argument('--force-recreate-vectorstore',
                        action='store_true',
                        help='Force recreation of vector store')
    parser.add_argument('--max-workers',
                        type=int,
                        default=DEFAULT_MAX_WORKERS,
                        help='Number of parallel workers')

    # Model options
    parser.add_argument('--llm-model',
                        default=DEFAULT_LLM_MODEL,
                        help='LLM model to use')
    parser.add_argument('--cache-file',
                        default='response_cache.pkl',
                        help='Path to cache file')

    # API key
    parser.add_argument('--api-key',
                        help='OpenAI API key (overrides env variable)')

    # Logging options
    parser.add_argument('--log-level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO',
                        help='Logging level')
    parser.add_argument('--log-file',
                        help='Log file path (default: auto-generated)')

    # Intermediate files
    parser.add_argument('--intermediate-dir',
                        default='intermediate',
                        help='Directory for intermediate files')

    return parser.parse_args()


def main():
    args = parse_args()  # Parse arguments

    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level=log_level, log_file=args.log_file)

    # Set API key if provided
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        logger.info("Using API key from command line arguments")

    # Run pipeline
    logger.info("Starting ISCO classification pipeline")
    success = run_pipeline(
        labeled_file_path=args.labeled_file,
        unlabeled_file_path=args.unlabeled_file,
        output_file_path=args.output_file,
        run_spellcheck=not args.skip_spellcheck,
        force_recreate_vectorstore=args.force_recreate_vectorstore,
        max_workers=args.max_workers,
        llm_model=args.llm_model,
        cache_file=args.cache_file,
        intermediate_output_dir=args.intermediate_dir
    )

    if success:
        logger.info("Pipeline completed successfully")
        return 0
    else:
        logger.error("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
