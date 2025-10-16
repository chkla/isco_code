"""
Configuration settings for the ISCO classification system.
"""

import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file if it exists
load_dotenv()

# API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Settings
DEFAULT_LLM_MODEL = "gpt-4.1"
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-large"
DEFAULT_TEMPERATURE = 0.0

# Vector Store Settings
VECTOR_STORE_PATH = "faiss_index_jobs"

# Processing Settings
DEFAULT_MAX_WORKERS = 5
DEFAULT_SIMILARITY_K = 7

# Cache Settings
DEFAULT_CACHE_FILE = "response_cache.pkl"

# RAG Prompt Templates
JOB_PROMPT_TEMPLATE = """
You are an expert in job classification. Your task is to find the most appropriate job titles (Berufsbezeichnung) for the given job description based on the information provided in the context.

Context (most similar job descriptions):
{context}

Job description to classify: {question}

Please select the top 3 most suitable job titles (Berufsbezeichnung) for this description. Provide a confidence score between 0 and 1 for each selection, where 1 is the highest confidence. Explain your reasoning for each selection.

Format your response as follows:
1. Code: [code], Berufsbezeichnung: [title], Confidence: [score]
   Reasoning: [Your explanation]

2. Code: [code], Berufsbezeichnung: [title], Confidence: [score]
   Reasoning: [Your explanation]

3. Code: [code], Berufsbezeichnung: [title], Confidence: [score]
   Reasoning: [Your explanation]

Please re-read carefully the job description you should classify!

Make sure to select exactly 3 options, even if you're less confident about the third one.
"""

# Spell checker prompt templates
SPELL_CHECK_TEMPLATE = """
You are a helpful and knowledgeable assistant for spell checking German professions according to DE-ISCO Classification.

Check the following German words which all should represent a profession and correct them if you found a spelling error (if the word is '-97 möchte ich nicht beantworten' then you can ignore this and return the same without change)

Word to check: {word}

Only return the corrected word without any additional text.

Corrected word:
"""

SPELL_CHECK_TEMPLATE_EXTENDED = """
You are a helpful and knowledgeable assistant for spell checking German professions according to DE-ISCO Classification.

Check the following German words which all should represent a profession and correct them if you found a spelling error (if the word is '-97 möchte ich nicht beantworten' then you can ignore this and return the same without change)

Context:
  - Person has completed vocational training (Berufsausbildung): {training}
  - Person has university degree (Universitätsabschluss): {unidegree}

Word to check: {word}

Only return the corrected word without any additional text.

Corrected word:
"""

# List of non-occupation terms for filtering
NON_OCCUPATION_TERMS = {
    'student', 'studentin', 'studenten', 'studentinnen',
    'schüler', 'schülerin', 'schülerinnen',
    'arbeitslos', 'arbeitslose', 'arbeitssuchend',
    'studiere', 'studierend', 'studierende', 'studierenden',
    'promovierende', 'promovend', 'promovendin', 'doktorand', 'doktorandin',
    'azubi', 'auszubildende', 'auszubildender',
    'rentner', 'rentnerin', 'pensionär', 'pensionärin',
    'elternzeit', 'mutterschutz', 'hausfrau', 'hausmann',
    'arbeitssuchende', 'arbeitsuchend', 'arbeitssuchend',
    'praktikant', 'praktikantin', 'angestellter', 'angestellte', 'angestellte*r'
    'selbstständiger', 'selbstständige', 'nichts',
    'arbeiter', 'mz brief', 'mitarbeiter', 'mitarbeiterin', 'keine',
    'die schule', 'abgestellte', '.', 'masterstudium', 'x', 'freiberufler', 'freiberuflich', 'master', 'abeiter', 'akademiker', 'xxx', 'werkstudent'
}


# Load blacklist from CSV file
def load_blacklist(file_path: str) -> set:
    df = pd.read_csv(file_path)

    return set(df['occupation'].str.lower().tolist())


# File paths (for testing and examples)
DEFAULT_PATHS = {
    "blacklist_file": "data/blacklist.csv",
    "labeled_file": "data/isco_occupations.csv",
    "unlabeled_file": "data/test/occupations_wave3.csv",
    "output_file": f"results/occupations-test-labeled-wave3-{DEFAULT_LLM_MODEL}.csv"
}

NON_OCCUPATION_TERMS.update(load_blacklist(DEFAULT_PATHS['blacklist_file']))
print(f"Loaded {len(NON_OCCUPATION_TERMS)} non-occupation terms from blacklist.")
