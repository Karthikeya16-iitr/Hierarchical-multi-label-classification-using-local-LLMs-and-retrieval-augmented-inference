# MavenMagnet NLP Classification Pipeline

A production-quality local NLP classification pipeline for predicting multiple valid Level 1–Level 2 associations for skincare/product reviews.

## Architecture

This project uses a standard `src`-layout structure:

```text
├── config/
│   └── config.yaml            # Pipeline configuration
├── data/
│   ├── input/                 # Original, untouched input workbooks
│   ├── interim/               # Intermediate parsed/split records
│   └── output/                # Formatted prediction workbooks
├── artifacts/                 # Saved models, vectorizers, metadata
├── reports/                   # Evaluation metrics and error analysis
├── logs/                      # Structured application logs
├── src/mavenmagnet/           # Modular package
│   ├── __init__.py            # Package root & logging initialization
│   ├── config.py              # Strongly-typed configuration schema (Pathlib/Pydantic/PyYAML)
│   ├── data_loader.py         # Input ingestion preserving source workbooks
│   ├── preprocessing.py       # Review normalization and tokenization
│   ├── taxonomy.py            # Canonical taxonomy definitions (43 valid pairs)
│   ├── dataset_builder.py     # Leak-free grouped train/eval splits
│   ├── retrieval.py           # Few-shot context selection
│   ├── llm_client.py          # Local Ollama client with schema validation & retry
│   ├── prompts.py             # Constrained prompt templates
│   ├── classifier.py          # TF-IDF + Logistic Regression baseline & LLM interfaces
│   ├── validators.py          # Strict prediction verification against taxonomy
│   ├── evaluator.py           # Exact match, Micro/Macro F1 evaluation
│   ├── exporter.py            # Formatted Excel exporter preserving source workbooks
│   └── pipeline.py            # Pipeline orchestrator
├── tests/                     # Unit and integration test suite
├── pyproject.toml             # Build system & pytest configuration
└── requirements.txt           # Pinned dependencies
```

## Setup & Installation

```bash
# Install package in editable mode with development dependencies
pip install -e .
```

## Running Tests

```bash
pytest
```
