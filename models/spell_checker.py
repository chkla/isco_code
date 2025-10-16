"""
Spell checker model for German profession descriptions.
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tenacity import retry, wait_exponential
from nltk.metrics import accuracy

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from utils.cache import ResponseCache
from config import SPELL_CHECK_TEMPLATE, SPELL_CHECK_TEMPLATE_EXTENDED, NON_OCCUPATION_TERMS

logger = logging.getLogger(__name__)


class GermanProfessionSpellChecker:
    """Spell checker for German profession descriptions."""

    def __init__(self, model_name: str = "gpt-4o",
                 temperature: float = 1,
                 use_education_context: bool = False,
                 cache: Optional[ResponseCache] = None):
        """
        Initialize the spell checker with specified model and temperature.

        Args:
            model_name: The OpenAI model to use
            temperature: Temperature setting for the model
            use_education_context: Whether to use educational background as context
            cache: Optional response cache
        """
        self.non_occupation_terms = NON_OCCUPATION_TERMS
        self.use_education_context = use_education_context
        self.cache = cache or ResponseCache()

        logger.info(f"Initializing spell checker with model {model_name}, "f"education context: {use_education_context}")

        self.llm = ChatOpenAI(
            model_name=model_name
        )

        if use_education_context:
            logger.info("Using education context for spell checking")
            self.template = SPELL_CHECK_TEMPLATE_EXTENDED
            self.input_variables = ["word", "training", "unidegree"]
        else:
            logger.info("Not using education context for spell checking")
            self.template = SPELL_CHECK_TEMPLATE
            self.input_variables = ["word"]

        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=self.template,
                input_variables=self.input_variables
            ),
            verbose=False
        )

    @retry(wait=wait_exponential(multiplier=1000, max=10000))
    def correct_word(self,
                     word: str,
                     training: Optional[bool] = None,
                     unidegree: Optional[bool] = None) -> str:
        """
        Correct a word with optional educational context.

        Args:
            word: The word to correct
            training: Whether the person has vocational training
            unidegree: Whether the person has a university degree

        Returns:
            The corrected word
        """
        if pd.isna(word) or word == "-97 möchte ich nicht beantworten":
            return word

        # Check cache first
        cache_key = f"{word}_{training}_{unidegree}" if self.use_education_context else word
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result.get('corrected', word)

        try:
            if self.use_education_context:
                result = self.chain.run(
                    word=word,
                    training=str(training),
                    unidegree=str(unidegree)
                ).strip()
            else:
                result = self.chain.run(word=word).strip()

            # Cache result
            self.cache.set(cache_key, {'corrected': result})
            return result

        except Exception as e:
            logger.error(f"Error correcting word '{word}': {e}")
            return word

    @retry(wait=wait_exponential(multiplier=1000, max=10000))
    def process_single_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single row with retry capability.

        Args:
            row: DataFrame row with occupation data

        Returns:
            Dictionary with processed data
        """
        try:
            result = row.to_dict()
            if self.use_education_context:
                result['corrected_occupation'] = self.correct_word(
                    word=row['occupation'],
                    training=row['training'],
                    unidegree=row['unidegree']
                )
            else:
                result['corrected_occupation'] = self.correct_word(
                    word=row['occupation']
                )
            return result
        except Exception as e:
            logger.error(f"Error processing row: {str(e)}")
            raise

    def batch_correct(self, words: List[str]) -> List[str]:
        """
        Correct a batch of words.

        Args:
            words: List of words to correct

        Returns:
            List of corrected words
        """
        return [self.correct_word(word) for word in words]

    def evaluate(self, test_df: pd.DataFrame) -> float:
        """
        Evaluate spell checker performance against a test set.

        Args:
            test_df: DataFrame with 'misspelling' and 'correct_spelling' columns

        Returns:
            Accuracy score
        """

        # Process words and calculate accuracy
        predictions = []
        references = test_df['correct_spelling'].tolist()

        for word in tqdm(test_df['misspelling'], desc="Processing words"):
            predictions.append(self.correct_word(word))

        acc = accuracy(references, predictions)
        logger.info(f"Spell checker accuracy: {acc:.2%}")
        return acc

    def process_dataframe(self, df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
        """
        Process a DataFrame with occupation descriptions.

        Args:
            df: DataFrame to process
            max_workers: Number of parallel workers

        Returns:
            Processed DataFrame
        """
        try:
            if 'occupation' not in df.columns:
                raise ValueError("DataFrame must contain a 'occupation' column")

            if self.use_education_context:
                edu_columns = ['training', 'unidegree']
                missing_columns = [col for col in edu_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Education context enabled but missing columns: {', '.join(missing_columns)}")

            # Process rows in parallel
            results = []
            logger.info(f"Processing {len(df)} rows with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.process_single_row, row) for _, row in df.iterrows()]

                for future in tqdm(futures, total=len(futures), desc="Processing rows"):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Failed to process row: {str(e)}")
                        continue

            # Convert results to DataFrame
            results_df = pd.DataFrame(results)

            # Calculate statistics
            total = len(results_df)
            changed = sum(results_df['occupation'] != results_df['corrected_occupation'])

            # Print summary
            logger.info("\nSpell check summary:")
            logger.info(f"Total entries processed: {total}")
            logger.info(f"Entries corrected: {changed}")
            logger.info(f"Percentage corrected: {(changed/total)*100:.2f}%")

            # Print example corrections
            if changed > 0:
                logger.info("\nExample corrections:")
                changes = results_df[
                    results_df['occupation'] != results_df['corrected_occupation']
                ].head(5)

                for _, row in changes.iterrows():
                    logger.info(f"Original: {row['occupation']}")
                    logger.info(f"Corrected: {row['corrected_occupation']}")
                    if self.use_education_context:
                        logger.info(f"Education: Vocational: {row['training']}, "f"University: {row['unidegree']}")
                    logger.info("-" * 50)

            return results_df

        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}")
            raise

    def is_valid_occupation(self, text: str) -> bool:
        """
        Check if the given text represents a valid occupation.

        Args:
            text: Text to check

        Returns:
            bool: False if text contains non-occupation terms or patterns
        """
        if pd.isna(text) or text == "-97 möchte ich nicht beantworten":
            return False

        # Convert to lowercase for comparison
        text_lower = text.lower()

        # Check for specific non-occupation terms
        words = text_lower.split()
        if any(word in self.non_occupation_terms for word in words):
            return False

        return True
