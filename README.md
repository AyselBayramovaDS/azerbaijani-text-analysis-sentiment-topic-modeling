# Azerbaijani Review Analysis: Sentiment Classification and Topic Modeling

An NLP pipeline built on about 159k real Azerbaijani app reviews. It answers two questions:
1. Is a review negative, neutral or positive?
2. What do users complain about and what do they praise?

The data is informal and noisy (misspellings, text without Azerbaijani letters, emojis, mixed languages, stars that do not always match the text, strong skew towards 5 stars), so part of the work is cleaning, defining labels, and evaluating honestly.

## Data

Hugging Face dataset `hajili/azerbaijani_review_sentiment_classification` (127,537 train and 31,885 test reviews; columns `content`, `score`, `upvotes`). It is loaded in the notebook with `datasets.load_dataset`, so no manual download is needed.

## Pipeline

1. **Data preparation:** data understanding, text cleaning, removal of empty and meaningless reviews, duplicate handling, test/train overlap check.
2. **Label definition:** 1-2 stars → negative, 3 → neutral, 4-5 → positive; stars checked against the text.
3. **Split and balancing:** train/validation split, balancing of the training part only.
4. **Baseline:** TF-IDF (character n-grams) + Logistic Regression.
5. **Transformer:** `xlm-roberta-base` fine-tuning for 3-class sentiment.
6. **Final evaluation:** test set, per-class metrics, confusion matrix, error analysis.
7. **Topic modeling:** BERTopic on LaBSE embeddings, separately for negative and positive reviews.
8. **Web demo:** Gradio interface for the saved model.

## Main results

| Macro-F1 | TF-IDF + LR | XLM-R |
|---|---|---|
| Validation | 0.560 | 0.559 |
| Test (full) | 0.578 | 0.580 |
| Test (new reviews only) | 0.554 | 0.565 |

Both models work reasonably well on negative and positive reviews and are weak on neutral. Topics: complaints are concrete (login and registration, payments and cards, app stability, taxi pricing, advertising), praise is general. Details and reasoning are in `note.md`.

## Repository contents

| File | Description |
|---|---|
| `textanalysis.ipynb`| The full pipeline in order |
| `note.md` | Detailed explanation of every stage: purpose, reasoning, results (English and Azerbaijani) |
| `README.md` | This file |
| `eval_test_report.csv` | Per-class precision, recall, F1 on the test set |
| `eval_confusion_matrix_test.png` | Test confusion matrix of XLM-R |
| `eval_comparison.csv` | Baseline vs XLM-R macro-F1 (full test and new reviews) |
| `topics_negative.csv`, `topics_positive.csv` | Topic sizes, shares and keywords |
| `train_clean.csv`, `test_clean.csv` | Cleaned data |
| `train_labeled.csv`, `test_labeled.csv` | Data with sentiment labels |
| `train_bal.csv`, `val.csv` | Balanced training set and validation set |
| `xlmr_sentiment/` | Saved XLM-R model and tokenizer |

## How to run

The notebook was developed on Kaggle.
1. Open the notebook and turn on a **GPU** and **Internet** (needed for model and dataset downloads, XLM-R fine-tuning and embeddings).
2. Install the extra packages that are not pre-installed: `pip install bertopic gradio` (BERTopic brings `umap-learn` and `hdbscan`).
3. Run the cells **from top to bottom, in order**. Later cells use variables created by earlier ones.
4. The CSV files, the evaluation files and the `xlmr_sentiment/` folder appear in the working directory.

Libraries used: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`, `datasets`, `transformers`, `torch`, `sentence-transformers`, `bertopic`, `gradio`.
