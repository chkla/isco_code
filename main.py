import os
import sys
from dotenv import load_dotenv

from utils.logging_config import setup_logging
from pipeline.runner import run_pipeline
from config import DEFAULT_PATHS


def main():
    """
    Main function to run the ISCO classification pipeline.

    Returns:
        0 for success, 1 for failure
    """

    load_dotenv()  # Load environment variables
    logger = setup_logging()  # Setup logging

    try:
        # Check for OpenAI API key
        if not os.environ.get('OPENAI_API_KEY'):
            logger.error("OPENAI_API_KEY environment variable not set")
            return 1

        # Run pipeline with default settings
        logger.info("Starting ISCO classification pipeline with default settings")

        success = run_pipeline(
            labeled_file_path=DEFAULT_PATHS['labeled_file'],
            unlabeled_file_path=DEFAULT_PATHS['unlabeled_file'],
            output_file_path=DEFAULT_PATHS['output_file']
        )

        if success:
            logger.info("Pipeline completed successfully")
            print(f"Results saved to: {DEFAULT_PATHS['output_file']}")
            return 0
        else:
            logger.error("Pipeline failed")
            print("Error: Pipeline failed. See logs for details.")
            return 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1

    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: An unexpected error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
