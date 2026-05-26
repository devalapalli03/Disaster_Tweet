# 📊 Results Summary — Disaster Tweets Classification

## Experiment Setup

- **Dataset**: Kaggle "Real or Not? NLP with Disaster Tweets"
- **Train size**: ~7,613 tweets | **Test size**: ~3,263 tweets
- **Class balance**: ~57% Non-Disaster, ~43% Disaster
- **Validation strategy**: 80/20 stratified split (random_state=42)
- **Primary metric**: F1 Score (binary)

---

## Model Comparison — Common Split

All models evaluated on the same 20% holdout split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| BERTweet (common) | ~0.83 | ~0.83 | ~0.81 | **~0.82** | ~0.88 | ~0.87 |
| DistilBERT (common) | ~0.80 | ~0.80 | ~0.79 | ~0.79 | ~0.86 | ~0.85 |
| LinearSVC char_wb 3–5g | ~0.79 | ~0.79 | ~0.76 | ~0.77 | ~0.84 | ~0.83 |
| LinearSVC word 1–2g | ~0.78 | ~0.77 | ~0.76 | ~0.76 | ~0.83 | ~0.82 |
| TF-IDF + LogReg | ~0.78 | ~0.77 | ~0.75 | ~0.76 | ~0.83 | ~0.82 |

---

## Model Comparison — Native Split

RNN models were evaluated on their own native split (saved artifacts).

| Model | F1 |
|-------|----|
| BiLSTM + Pooling (native) | ~0.74 |
| SimpleRNN (native) | ~0.70 |

---

## Best Model: BERTweet

**Why BERTweet wins:**
- Pre-trained on **850M English tweets** — understands hashtags, @mentions, emojis, slang out of the box
- Fine-tuned with prompt-style inputs combining `TEXT`, `KEYWORD`, and `LOCATION` metadata
- Best F1 after threshold sweep (~0.82 at optimized threshold)
- Outperforms DistilBERT by ~3 F1 points on this task

---

## Preprocessing Strategies Tested

| Strategy | Models Used |
|----------|-------------|
| Light clean (preserve hashtags/emojis, replace URLs/@mentions with `HTTPURL`/`@USER`) | BERTweet, DistilBERT |
| Heavy clean (contractions, lemmatization, stop-word removal, emoji strip) | RNN, BiLSTM, TF-IDF, LinearSVC |
| Prompt-style inputs (`TEXT: ...; KEYWORD: ...; LOCATION: ...`) | BERTweet only |
| Fill missing keyword/location with placeholder tokens | All transformer models |

---

## Error Analysis

### False Positives (predicted disaster, actually not)
- Metaphors: *"my phone is on fire"*, *"she killed it on stage"*
- Hyperbole: *"this traffic is a disaster"*

### False Negatives (predicted non-disaster, actually disaster)
- Sparse context: *"evacuating now"*, *"help needed"*
- Ambiguous or location-free tweets

### Mitigation Ideas
- Add disaster keyword lexicons as features
- Data augmentation (back-translation, paraphrasing)
- BERTweet + char-SVC soft ensemble
- Re-weigh samples near the decision boundary

---

## Threshold Tuning

LinearSVC uses `decision_function` scores (no native probabilities). We swept thresholds from the 5th–95th percentile of scores to maximize F1:

- **Word 1–2-grams**: best threshold varies by run (~0.1–0.4)
- **Char 3–5-grams**: best threshold varies by run (~0.1–0.4)

BERTweet and DistilBERT use softmax probabilities — threshold was kept at 0.5 for the common-split comparison (and tuned for best-F1 in the standalone BERTweet cell).

---

## Kaggle Submission

Final submission generated using fine-tuned BERTweet:
- File: `submission_bertweet.csv`
- Format: `id`, `target` (0 or 1)
- Threshold: 0.50

---

## Reproducibility Notes

All splits use `random_state=42` and `stratify=y`. Model checkpoints are saved to:
- `./bt_results/` — BERTweet (HuggingFace Trainer checkpoints)
- `/content/dataset/distilbert_finetuned/` — DistilBERT (TF SavedModel)
- `/content/dataset/artifacts/` — RNN/BiLSTM `.h5` files + `.npy` val arrays

To reproduce: run all notebook cells in order within a GPU-enabled Colab session.
