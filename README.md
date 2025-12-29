# Fake vs Real Job Postings – NLP Capstone

## Problem
Online job platforms are increasingly targeted by fraudulent postings, which
can mislead job seekers and cause financial or personal harm.  
The goal of this project was to build and evaluate machine learning models
capable of classifying job postings as **real or fake** using both textual
and structured features.

---

## Data
The dataset combines job postings from multiple sources and includes:

- **Text features:** job descriptions, requirements, and benefits
- **Structured features:** location, salary information, and job metadata

The data required extensive preprocessing to handle missing values,
inconsistent formatting, and noisy text.

---

## Approach
This project followed an end-to-end NLP and machine learning pipeline:

- Text cleaning and feature extraction
- Baseline models:
  - Naive Bayes
  - Logistic Regression
- Advanced models:
  - Support Vector Machines
  - Decision Trees
- Ensemble methods to improve robustness
- Model evaluation using **precision, recall, and F1-score**, with particular
  emphasis on recall for fraudulent postings

---

## Results & Insights
- Baseline models provided strong interpretability and fast iteration
- More complex models improved performance on nuanced language patterns
- Ensemble approaches increased overall robustness compared to individual models
- Recall for fraudulent postings was prioritized to reduce false negatives

---

## Tools & Technologies
- Python
- pandas, NumPy
- scikit-learn
- Natural Language Processing (NLP) techniques

---

## Key Takeaways
This project highlights practical challenges in real-world NLP workflows,
including noisy text data, class imbalance, and tradeoffs between model
performance and interpretability.
