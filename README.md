# Transformer-Based LLM Framework for Automated Feature Extraction and Context-Aware Data Mining

This repository contains a research-grade Python project designed to demonstrate the efficacy of Large Language Models (LLMs) in automating feature extraction and improving contextual understanding compared to traditional data mining techniques.

## Research Motivation

Traditional data mining often relies on lexical and statistical feature extraction such as TF-IDF, which ignores semantic context and word order. This framework evaluates how transitioning to contextualized embeddings from pre-trained Transformers (like DistilBERT) enhances performance on unstructured text data tasks, specifically sentiment analysis.

## Architecture Explanation

The framework is broken down into modular components to ensure high cohesion and low coupling:

- **`src/data_preprocessing.py`**: Handles text cleaning and standardizing input data.
- **`src/traditional_pipeline.py`**: Implements baseline models using TF-IDF and Logistic Regression.
- **`src/llm_pipeline.py`**: Tokenizes text and extracts semantic embeddings using an LLM.
- **`src/model_training.py`**: Orchestrates train-test splits and cross-validation cleanly.
- **`src/evaluation.py`**: Calculates classification metrics and execution efficiency.

## Installation Steps

1. Clone this repository or download the source code.
2. Ensure you have Python 3.9+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Instructions

1. Place your dataset inside the `data/` directory named `dataset.csv`. The dataset must contain `text` and `label` columns.
2. Run the main experiment script from the root directory:
   ```bash
   python main.py
   ```

## Experimental Methodology

The framework automatically performs the following:

1. **Data Preprocessing**: Cleans input text (URLs, emojis, lowercasing).
2. **Pipeline Execution**:
   - **Baseline**: Trains a Logistic Regression model on TF-IDF features.
   - **LLM-Based**: Extracts dense contextual features via DistilBERT and trains a linear classifier on top.
3. **Evaluation**: Compares the models based on Accuracy, Precision, Recall, F1-score, and Training Time.

## Reproducibility Notes

Reproducibility is enforced globally via specific seed constants defined in `src/config.py`. Using `utils.set_seed()` guarantees deterministic train-test splits, model initializations, and inference across experiments.
