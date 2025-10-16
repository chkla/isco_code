# Automating Occupational Coding in Survey Research: A Retrieval-Augmented LLM Approach

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

This repository contains the implementation of an automated pipeline for occupational coding using Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). The system automatically identifies and assigns ISCO (International Standard Classification of Occupations) codes to occupation descriptions from survey data, with built-in spell-checking support for non-native speakers responses.

## ✨ Key Features

- **Automated ISCO Coding**: Automatically assigns ISCO codes to occupation descriptions
- **RAG Architecture**: Leverages retrieval-augmented generation for improved accuracy
- **Spell Checking**: Built-in support for handling typos and non-native speaker variations
- **Flexible Configuration**: Easy-to-customize pipeline parameters
- **Explainable Predictions**: Provides model explanations alongside predictions
- **Vector Database Integration**: Efficient similarity search using FAISS

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT models)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/occupational-coding-rag.git
cd occupational-coding-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running the Pipeline

1. Configure your settings in `config.py`:
```python
DEFAULT_LLM_MODEL = "gpt-4-turbo"
DEFAULT_EMBEDDINGS_MODEL = "text-embedding-3-large"
VECTOR_STORE_PATH = "faiss_index_jobs"
```

2. Run the pipeline:
```bash
python main.py
```

3. Find your results in the `results/` folder.

## ⚙️ Configuration

The pipeline can be customized through `config.py`:

```python
DEFAULT_PATHS = {
    "blacklist_file": "data/blacklist.csv",        # Occupation descriptions to ignore (e.g., "student")
    "labeled_file": "data/isco_occupations.csv",   # Pre-labeled occupation descriptions with gold ISCO codes
    "unlabeled_file": "data/test/occupations_wave3.csv",  # New unlabeled file for processing
    "output_file": f"results/occupations-test-labeled-wave3-{DEFAULT_LLM_MODEL}.csv"  # Output file
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DEFAULT_LLM_MODEL` | LLM model for generation | `gpt-4-turbo` |
| `DEFAULT_EMBEDDINGS_MODEL` | Embedding model for retrieval | `text-embedding-3-large` |
| `VECTOR_STORE_PATH` | Path to vector database | `faiss_index_jobs` |
| `BLACKLIST_FILE` | Invalid occupation descriptions | `data/blacklist.csv` |
| `LABELED_FILE` | Gold standard ISCO mappings | `data/isco_occupations.csv` |
| `UNLABELED_FILE` | Input file for coding | `data/test/occupations_wave3.csv` |

## 📁 Project Structure

```
occupational-coding-rag/
│
├── data/
│   ├── blacklist.csv           # Occupation descriptions to exclude
│   ├── isco_occupations.csv    # Labeled training data
│   └── test/                   # Test datasets
│   └── training/               # Training datasets
│   └── raw_files/              # Offical ISCO files
│
├── results/                    # Output directory for predictions
├── models/                     # Modeling scripts
├── faiss_index_jobs/           # Vector database storage
├── utils/                      # Helper files
│
├── main.py                     # Main pipeline script
├── cli.py                      # Command-Line-Interface script helper
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 📊 Input/Output Format

### Input Format
The unlabeled CSV file should contain occupation descriptions in the following format:
```csv
id_occ,unidegree,occupation
5,0,Krankenschwester 
6,1,Kinderbetreuerin
15,1,Sozial
22,1,Sozialarbeiterin 
26,0,Bauzeichnerin
53,1,Apothekerin 
57,1,Sozial Arbeiterin 
67,0,Physiotherapeutin 
79,1,Engineering 
85,1,MFA
```

### Output Format
The pipeline generates a CSV with ISCO predictions and explanations:
```csv
id_occ,unidegree_x,occupation_original,unidegree_y,occupation_corrected,predicted_code_1,predicted_berufsbezeichnung_1,confidence_1,..."
5,0,Krankenschwester,no,Krankenschwester,3221,Krankenpflegerin,0.95,"""Krankenschwester"" is the traditional term for a female nurse..."
6,1,Kinderbetreuerin,yes,Kinderbetreuerin,5311,Kinderbetreuer,0.95,"The job description ""Kinderbetreuerin"" is the direct feminine form of..."
```

## 📖 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{xxx,
    author = {xxx},
    year = {2025},
    title = {Automating Occupational Coding in Survey Research: A Retrieval-Augmented LLM Approach},
    DOI = {xxx}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

Made with ❤️ and 🤖
