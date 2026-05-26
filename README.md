# 🌪️ Disaster Tweets Classification

> **NLP with Disaster Tweets** — Kaggle Competition Project  
> *Real or Not? Classifying tweets as disaster-related or non-disaster*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-BERTweet%20%7C%20DistilBERT-yellow)](https://huggingface.co)
[![Kaggle](https://img.shields.io/badge/Kaggle-NLP%20Disaster%20Tweets-20BEFF?logo=kaggle)](https://www.kaggle.com/c/nlp-getting-started)

---

## 👥 Team

| Name | Role |
|------|------|
| Jonathan James | Deep Learning, Transformer Fine-tuning |
| Lakshmi Narasimha Koushik Devalapalli | NLP, Classical ML, EDA |

---

## 📌 Project Overview

This project tackles the [Kaggle "Real or Not? NLP with Disaster Tweets"](https://www.kaggle.com/c/nlp-getting-started) competition. The goal is to build a binary text classifier that distinguishes real disaster tweets from non-disaster ones.

We trained and compared **6 model families** — from classical TF-IDF baselines to fine-tuned Transformer models:

| Model | Type | Best F1 |
|-------|------|---------|
| **BERTweet** | Transformer (Twitter-pretrained) | **~0.82** ✅ |
| DistilBERT | Transformer | ~0.79 |
| LinearSVC (char n-grams) | Classical ML | ~0.77 |
| TF-IDF + Logistic Regression | Classical ML | ~0.76 |
| BiLSTM + Pooling | Deep Learning | ~0.74 |
| SimpleRNN | Deep Learning | ~0.70 |

**Winner: BERTweet** — pretrained on Twitter data, making it the best fit for hashtags, slang, emojis, and short noisy context.

---

## 📂 Repository Structure

```
disaster-tweets-classification/
│
├── disaster_tweets_classification.ipynb  # 📓 Main notebook (full pipeline)
├── presentation.pptx                     # 📊 Project presentation slides
│
├── README.md                             # You're here
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Files to exclude from version control
│
└── docs/
    └── results_summary.md               # Model comparison & key findings
```

> **Dataset:** Download from [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data) and place as `nlp-getting-started.zip` before running the notebook.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/disaster-tweets-classification.git
cd disaster-tweets-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

Download `nlp-getting-started.zip` from [Kaggle](https://www.kaggle.com/c/nlp-getting-started/data) and place it in the project root.

### 4. Run the notebook

Open `disaster_tweets_classification.ipynb` in **Google Colab** (recommended for GPU access) or Jupyter:

```bash
jupyter notebook disaster_tweets_classification.ipynb
```

> ⚠️ For GPU-accelerated Transformer training, **Google Colab (T4/A100)** or a CUDA-enabled machine is strongly recommended.

---

## 🗂️ Notebook Structure

The notebook is organized into 5 sections:

### 1. Introduction
- Project overview and objectives
- Dataset description (id, keyword, location, text, target)

### 2. Data Loading, Exploration & Preprocessing
- **EDA**: Class distribution, keyword/location analysis, word clouds, n-grams, length distributions
- **Advanced cleaning**: URL/mention normalization, contraction expansion, emoji removal, lemmatization
- **Feature engineering**: Sentiment polarity, word/char counts, URL and location presence flags
- **Tokenization & padding** for RNN/LSTM models (vocab=20k, maxlen=50)
- **BERTweet-friendly prep**: Light cleaning that preserves hashtags/emojis + prompt-style inputs

### 3. Model Building & Training
- `3.1` SimpleRNN
- `3.2` BiLSTM with Pooling
- `3.3` TF-IDF + Logistic Regression
- `3.4` LinearSVC (word & character n-grams, GridSearch)
- `3.5` DistilBERT fine-tuning (TensorFlow, warmup + weight decay)
- `3.6` BERTweet fine-tuning (PyTorch via HuggingFace `Trainer`)

### 4. Model Evaluation & Comparison
- Unified comparison across all models
- Native split (RNN/BiLSTM) vs. Common split (everyone else)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Threshold tuning for best F1 (especially LinearSVC)
- ROC and Precision-Recall curve visualizations

### 5. Kaggle Submission
- BERTweet inference on test set
- Outputs `submission_bertweet.csv`

---

## 📊 Key Results

### Model Comparison (Common Split)

| Model | F1 | ROC-AUC |
|-------|----|---------|
| BERTweet | **~0.82** | **~0.88** |
| DistilBERT | ~0.79 | ~0.86 |
| LinearSVC (char 3–5-grams) | ~0.77 | ~0.84 |
| LinearSVC (word 1–2-grams) | ~0.76 | ~0.83 |
| TF-IDF + LogReg | ~0.76 | ~0.83 |

### Native Split (RNN Models)

| Model | F1 |
|-------|----|
| BiLSTM + Pooling | ~0.74 |
| SimpleRNN | ~0.70 |

---

## 🔍 Insights & Error Analysis

- **Metaphors/sarcasm** (e.g., *"my phone is on fire"*) → false positives
- **Sparse context** (e.g., *"evacuating now"*) → some false negatives
- **Char n-grams** are competitive for handling slang, misspellings, and hashtags
- **BERTweet** wins because it was pretrained on ~850M English tweets — the domain match matters

### Potential Improvements
- Keyword lexicons for disaster-specific terms
- Data augmentation (back-translation, synonym swap)
- Ensemble: BERTweet + char-SVC
- Extended training with cosine-annealing schedule

---

## 🌍 Social Impact

This classifier enables:
- **Faster situational awareness**: Flag disaster tweets in near-real time for emergency operations
- **Better resource allocation**: Prioritize locations/keywords tied to real incidents
- **Information quality**: Surface credible disaster reports, filter benign noise
- **Edge deployment**: Lightweight classical models (SVC/LogReg) run on low-cost servers; BERTweet for high-accuracy back-end verification

---

## 🛠️ Tech Stack

- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, wordcloud
- **NLP/Preprocessing**: NLTK, contractions, emoji, TextBlob
- **Classical ML**: scikit-learn (TF-IDF, LogisticRegression, LinearSVC)
- **Deep Learning**: TensorFlow/Keras (RNN, BiLSTM, DistilBERT)
- **Transformers**: HuggingFace `transformers`, `datasets`, `accelerate`
- **Models**: `distilbert-base-uncased`, `vinai/bertweet-base`

---

## 📄 License

This project is for academic/educational purposes as part of a Deep Learning course.

---

## 🙏 Acknowledgements

- [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
- [VinAI Research — BERTweet](https://github.com/VinAIResearch/BERTweet)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
