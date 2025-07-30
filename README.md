# SMS Spam Detection – NLP Pipeline

## Project Overview
This project implements an end-to-end NLP pipeline to classify SMS messages as spam or ham (not spam). The pipeline uses classical machine learning techniques, including:

- Text preprocessing (tokenization, lemmatization, stopword removal)
- Sparse and dense feature engineering (BoW, TF-IDF, Word2Vec)
- Generative and discriminative classifiers (Naive Bayes, Logistic Regression)
- Evaluation using precision, recall, accuracy, and F1-score

## Use Case
**Stakeholder**: Telecom providers  
**Problem**: Automatically detect and filter spam messages to protect users from fraud and improve service quality.

## Dataset
- Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Format: TSV with labels (`ham`/`spam`) and message text

## Features Used
- **Bag-of-Words (BoW)** – CountVectorizer
- **TF-IDF** – TfidfVectorizer
- **Word2Vec** – Pre-trained GloVe embeddings (100-dim)

## Models & Evaluation
| Model               | Feature   | Accuracy | F1-score |
|--------------------|-----------|----------|----------|
| Naive Bayes         | BoW       | ~0.97    | ~0.96    |
| Logistic Regression | TF-IDF    | ~0.98    | ~0.97    |
| Logistic Regression | Word2Vec  | ~0.95    | ~0.94    |

## Reproducibility
- All random seeds set (`random`, `numpy`)
- Modular code structure in one script
- Easy to reproduce in a Jupyter Notebook or `.py` script

## Requirements
See `requirements.txt` for all necessary libraries:
- pandas
- numpy
- nltk
- scikit-learn
- gensim

## Conclusion
This NLP pipeline demonstrates the effectiveness of classical NLP techniques for real-world spam detection. Logistic Regression with TF-IDF gave the best balance of performance and interpretability.
