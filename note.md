# Azerbaijani Review Analysis: Sentiment Classification and Topic Modeling

## 1. Goal and data

**Goal.** Using about 159k real Azerbaijani app reviews, answer two questions: (1) is a review negative, neutral or positive, and (2) what do users complain about and what do they praise.

**Data.** Hugging Face dataset `hajili/azerbaijani_review_sentiment_classification`: 127,537 train and 31,885 test reviews with columns `content` (review text), `score` (1-5 stars) and `upvotes`. `upvotes` was not used.

**Why the work is organized this way.** The data is informal and noisy: misspellings, text written without Azerbaijani letters (`cox` instead of `çox`), emojis, mixed languages, stars that do not always match the text, and a strong skew towards 5 stars. So the order is: understand the data → clean it → define labels → split and balance → baseline model → transformer model → evaluate → find topics → demo. Every decision below is based on what was seen in the data at the previous step.

## 2. Data understanding

**Purpose.** Decide on cleaning, labeling and evaluation by looking at the data first.

**What was done.** Checked sizes, missing values, star distribution, review length, exact text duplicates, and read sample reviews.

**Results.**
- Train has 3 empty `content` values; there are no other missing values.
- Stars are strongly skewed: 5 stars 81.4%, 1 star 11.3%, and 2-3-4 stars together only 7.3%. Consequences: the training set must be balanced, and accuracy alone cannot be trusted.
- Reviews are very short: median 10 characters, mean 30, 75% shorter than 34 characters, maximum 1,987. Consequences: a maximum length of 128 tokens is enough for the transformer, and short reviews such as `Əla` or `hə` must not be removed by a length rule.
- 62,850 train reviews have exactly the same text as an earlier review, so duplicates need a separate decision (Section 5).

## 3. Text cleaning

**Purpose.** Reduce noise without destroying information that carries sentiment.

**What was done.** The function `clean()` applies three rules in this order and writes the result into a new `text` column (the original `content` is kept):
1. remove URLs (done first, because the next rule would turn `www` into `ww` and the link would no longer be recognized);
2. shorten any character repeated 3 or more times to 2 (`yaxşıııı` → `yaxşıı`; not to 1, because the stretching is itself an emotional signal);
3. collapse extra whitespace (done last, because removing a URL leaves extra spaces).

**What was deliberately kept:** emojis (they carry sentiment), letter case, and common words (the transformer reads context, so removing words like `deyil` could flip the meaning). Python's `.lower()` was not used because it handles the Azerbaijani letters `I/ı` and `İ/i` incorrectly.

The same function is applied to train and test, so the model always sees the same text format.

**Result.** Sample comparisons of `content` and `text` show that only noise changed (`Ellaaaa` → `Ellaa`, `✅✅✅` → `✅✅`) and the meaning is preserved.

## 4. Removing empty and meaningless reviews

**Purpose.** A review with no content cannot teach or test sentiment.

**What was done.** A review is kept if the cleaned text contains a letter, a digit or any non-ASCII character (emoji, `ə`, `ç`, ...). Before deleting, the rows to be removed were listed and checked.

**Result.** 37 train and 11 test reviews were removed: empty texts and texts such as `.`, `..`, `??`, `@`, plus a few ASCII emoticons like `:)`. Short real answers (`hə`, `ok`, `👍`) were kept. After this step: 127,500 train and 31,874 test reviews, with no missing values.

## 5. Duplicate reviews in train

**Purpose.** Prevent the model from memorizing frequent phrases and prevent the same text from appearing in both train and validation.

**What was done.** Duplicates were counted on the cleaned `text` (not on `content`), because the model only sees `text`, and cleaning makes different spellings identical (`Superrrr` and `Superr`). They were counted only by text: `score` and `upvotes` are not model inputs, so a different upvote count does not make a different example.

**Findings.** 63,602 train reviews (49.9%) repeat an earlier text. The most frequent are `Əla` (13,288), `Super` (6,980), `👍` (4,292) and `👍👍` (2,154). These are real short answers from different users, not data errors, so removing them is a modeling decision, not a data fix.

**Decision.** In train, each text is kept once (63,898 reviews left, 0 duplicates). The test set was not changed, so it keeps the real distribution.

**Effect.** The share of 5-star reviews fell from 81.4% to 68.9% and the share of 1-star reviews rose from 11.3% to 20.3%. Imbalance decreased but remains.

## 6. Overlap between test and train

**Purpose.** Because the test set was not deduplicated, check how many test reviews are already present in train.

**What was done.** Each test review was flagged in a column `seen_in_train` (nothing was removed): `True` if its text exists in the cleaned train split.

**Results.** 16,592 test reviews (52.1%) are seen; 15,282 (47.9%) are new. The overlap is mostly `Əla` (3,317), `Super` (1,676), `👍` (1,042). The star distribution of the new reviews (68.2% five stars, 20.8% one star) is very close to the deduplicated train, while the full test set has 81.4% five stars.

**Consequence.** Results on the full test set can look better than they are, so evaluation is reported in two parts: the full test set and the new reviews only. (`seen_in_train` is computed against the full cleaned train split, not only against the smaller balanced subset used for training.)

## 7. Label definition

**Rule.** 1-2 stars → negative, 3 → neutral, 4-5 → positive. The model learns classes, not stars. This is the most common and easiest rule to explain: 1-2 is dissatisfaction, 3 is a mixed opinion, 4-5 is satisfaction. The same rule is used for train and test.

**Class distribution.** Train: positive 46,954 (73.5%), negative 14,678 (23.0%), neutral 2,266 (3.5%). Test: positive 85.1%, negative 12.8%, neutral 2.1%. An "always positive" model would reach about 85% accuracy on test while learning nothing, so macro-F1 and per-class metrics are used.

**Do stars match the text?** The task asks to inspect this, so four checks were made:
1. *10 random reviews per class were read.* Negative reviews are clear complaints (login, SMS, app not opening). Neutral reviews are mostly mild complaints or mixed opinions (`Pis deyil amma...`, `Biraz düzəliş olsa daha gözəl olar`). Positive reviews are mostly praise, with an exception (`Gencol tarifinden hec razi qalmadim`, 5 stars).
2. *The most frequent words per class were listed.* Negative: `pis`, `olmur`, `pul`, `qeydiyyatdan`; neutral: `amma`, `olar`, `olmasa`, `olsa`; positive: `əla`, `super`, `gözəl`, `rahat`. Writing without Azerbaijani letters is very common (`çox` 8,335 vs `cox` 5,142 in positive reviews).
3. *Reviews consisting only of digits were checked*, because the token `1` was frequent in negative reviews (to see whether users write their star in the text). Only 3 such reviews exist, so there is no direct leak of the star into the text.
4. *A rough keyword filter* (word lists chosen by hand from the frequent-word output) looked for negative words in positive-labeled reviews (1,618) and positive words in negative-labeled reviews (385). It is only a way to find suspicious cases, not an exact measure, because it also catches negations (`pis deyil`) and conditional phrases (`əla olardı`).

**Reading the suspicious cases showed two kinds of contradictions:** praise with 1-2 stars (`Əla proqramdır 👍👍`, 2 stars; `Super😍`, 1 star), and users who describe a problem but give 4-5 stars (`Birbank niyə işləmir`, 5 stars). Some matches were filter errors or mixed texts.

**Decision.** Labels are kept as given by the stars. Reasons: the star is the user's own rating; correcting labels with word rules would introduce new errors (negation, conditionals); changing test labels would make the evaluation unreliable. The label noise that remains is taken into account when reading the results (Sections 12-13).

## 8. Train / validation split

**Purpose.** Training needs a data part to monitor progress and choose the best epoch. If the test set were used for this, the final result would be inflated.

**What was done.** The cleaned, deduplicated train (63,898) was split 85% / 15% into `train_part` (54,313) and `val_df` (9,585), stratified by label with a fixed random seed. The test set (31,874) was not touched.

**Why this is safe.** Duplicates were removed before the split, so the same text cannot be in both parts. Validation keeps the natural class distribution (73.5% / 23.0% / 3.5%), so model selection is based on realistic data. Neutral has 1,926 reviews in `train_part` and 340 in validation.

## 9. Balancing the training part

**Purpose.** On imbalanced data a model can look good by always predicting the majority class. Balancing forces it to learn all three classes.

**What was done.** Only `train_part` was balanced: up to 6,000 random reviews for negative and for positive, and all 1,926 neutral reviews, giving `train_bal` with 13,926 reviews. The result was saved to files.

**Trade-off.** Many reviews are dropped and the real class frequencies are changed (neutral is still about 3 times smaller than the others). This is why validation and test were left in their natural distribution.

## 10. Baseline model: TF-IDF + Logistic Regression

**Purpose.** A simple, fast reference point to judge whether the transformer really adds value.

**What was done.** Text was converted to TF-IDF vectors on character n-grams of length 2-5 (within word boundaries), 45,827 features. Character pieces were chosen because the data has many spelling variants (`çox/cox`, `əla/ela`), and such variants share many character pieces. The vectorizer was fitted only on `train_bal` and only applied to validation, so no information leaks. Logistic Regression was trained with `class_weight="balanced"` to compensate for the remaining imbalance.

**Validation result (9,585 reviews, natural distribution).** Macro-F1 0.560; accuracy 0.735.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.662 | 0.719 | 0.689 |
| neutral | 0.089 | 0.426 | 0.147 |
| positive | 0.955 | 0.755 | 0.843 |

Accuracy equals the accuracy of an "always positive" model (73.5%), which confirms that macro-F1 is the metric to use. Neutral is weak.

**Sanity check.** The most influential character pieces per class make sense: negative `mir` (from `işləmir`, `açılmır`), `pis`, 👎; positive `əla`, `ela`, 👍; neutral `eyil` (from `deyil`) and `?`. This shows the model learned meaningful patterns and that emojis carry real signal (so keeping them was right).

## 11. Transformer model: XLM-R fine-tuning

**Purpose.** A pretrained multilingual transformer understands context (for example the difference between `pis` and `pis deyil`), so it is expected to handle sentiment better than character statistics.

**What was done.**
- Texts were tokenized with the `xlm-roberta-base` tokenizer (maximum length 128, dynamic padding). Misspelled words are split into smaller pieces (for example `Mukəmməl` → `Muk`, `əm`, `mə`, `l`), so the model can still extract meaning, which is why heavy cleaning was not needed.
- The pretrained model received a new 3-class output layer (the warning about newly initialized `classifier` weights is expected).
- Training: `train_bal`, 3 epochs, learning rate 2e-5, batch size 32 per device, mixed precision (fp16), fixed seed. After each epoch the model was evaluated on validation, and the best epoch was selected by **validation macro-F1** (not accuracy, which is misleading here).

**Training progress.** Validation macro-F1 per epoch: 0.514 → 0.521 → 0.559, still increasing at epoch 3. Training took about 11 minutes.

**Validation result.** Macro-F1 0.559, accuracy 0.802.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.595 | 0.877 | 0.709 |
| neutral | 0.099 | 0.088 | 0.093 |
| positive | 0.949 | 0.813 | 0.876 |

**Comparison with the baseline on validation.** XLM-R is better on negative (0.709 vs 0.689) and positive (0.876 vs 0.843) but worse on neutral (0.093 vs 0.147), so macro-F1 is practically the same (0.559 vs 0.560).

## 12. Final evaluation on the test set

**Purpose.** The test set was not used for training or for any choice, so it gives the most honest result.

**What was done.** The saved XLM-R model was evaluated on all 31,874 test reviews, then separately on the seen and unseen parts (Section 6), and the baseline was measured on exactly the same data.

**Results on the full test set.** Macro-F1 0.580, accuracy 0.887.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| negative | 0.594 | 0.834 | 0.694 |
| neutral | 0.129 | 0.091 | 0.107 |
| positive | 0.967 | 0.915 | 0.940 |

**Two parts of the test set.**

| Part | Reviews | Accuracy | Macro-F1 | F1 negative / neutral / positive |
|---|---|---|---|---|
| Seen in train | 16,592 | 0.971 | 0.521 | 0.578 / 0.000 / 0.985 |
| New reviews | 15,282 | 0.797 | 0.565 | 0.705 / 0.119 / 0.870 |

Accuracy is much higher on the seen part because it is almost entirely short positive phrases; there neutral is never found. The new reviews are the more realistic measure of how the model works on unseen text.

**Baseline vs XLM-R (macro-F1).**

| Data | Baseline | XLM-R |
|---|---|---|
| Validation | 0.560 | 0.559 |
| Test, full | 0.578 | 0.580 |
| Test, new reviews | 0.554 | 0.565 |

**Interpretation.** The two models perform almost the same: the transformer did not clearly beat the simple baseline in this setup. It is better on negative and positive, but neutral stays weak in both models. The confusion matrix shows that neutral is mixed with both neighboring classes, and that a noticeable part of positive reviews is predicted as negative, which lowers the precision of the negative class. This is consistent with what was seen when reading the neutral reviews (mild complaints, close to negative) and with the label noise found in Section 7.

## 13. Error analysis

**Purpose.** Understand what kind of mistakes the model makes.

**What was done.** The 3,589 test errors were sorted by the model's confidence, and the 20 most confident errors were read.

**Result.** All 20 are clear praise texts with 1-3 stars (`Əla 👍` with 1 star, `Super 👍 👍 👍 👍`, `Çox qəşəng` with 2-3 stars), which the model read as positive. In these cases the model is right and the label is wrong. So part of the "errors", and therefore part of the weakness in the test scores, comes from label noise, and the test metrics probably understate the model's real quality. Labels were not changed (decision in Section 7).

## 14. Topic modeling

**Purpose.** Sentiment tells whether users are satisfied; topics tell what they talk about. This answers the second question: what users complain about and what they praise.

**Data selection.** From the cleaned train, only reviews with at least 4 words were used (very short reviews such as `Əla` or `👍` do not form meaningful topics), and up to 10,000 random reviews were taken for negative and for positive separately. Separate models answer two separate questions: complaints and praise.

**Embeddings.** Each review was converted into a 768-dimensional vector with the multilingual LaBSE model (used as it is, not trained). Reviews with similar meaning get similar vectors, and topic grouping is based on this similarity.

**BERTopic.** Vectors are reduced with UMAP, grouped with HDBSCAN (a topic must contain at least 60 reviews; the number of topics is not fixed in advance), and keywords are extracted for each topic. Settings: `language="multilingual"` so that Azerbaijani letters are kept in the keywords; a small stopword list prepared by hand from the frequent-word output, because there is no ready Azerbaijani list (meaningful words such as `pis`, `olmur` were not included in it). The settings are starting values; the random seed is fixed.

**Reading the topics.** The topic tables give sizes and keywords. To name topics reliably, 3 real reviews from each of the 5 largest topics of each sentiment were read; the remaining topics are described by keywords only.

**Negative reviews: 14 topics.** 3,380 reviews (33.8%) do not belong to any topic (normal for short noisy texts). The topics:
- Topic 0 (3,120 reviews, the largest and a broad one): login and registration errors: number or code not accepted, error messages, app not opening.
- Topic 1: reviews starting with `Salam` with varied content (taxi order refusals, deleting a profile, questions). This is a writing-style group, not a content topic.
- Topic 2: taxi service: routes, prices, cashback, finding the car (keywords `taksi`, `manat`, `sifariş`).
- Topic 3: "why" questions: payment or card transfer not possible, app freezing, not loading.
- Topic 4: advertising and spam messages (keywords `reklam`, `elan`).
- Smaller topics (keywords only): internet packages, server not responding, card and money, a specific bank app, "it used to be good, now it is not".

**Positive reviews: 25 topics.** 3,481 reviews (34.8%) have no topic. The topics:
- Topic 0 (1,951 reviews, the largest): despite the positive stars, these reviews contain questions and problems (money deducted from balance, app not working, card questions).
- Topic 1: suggestions (`olsa yaxşı olar`).
- Topic 2: general praise of the app.
- Topic 3: using the app for work and job ads.
- Topic 4: praise of a bank service and its staff.
- Other topics (keywords only): convenience and low price (`rahat`, `asan`, `sərfəli`), thanks (`sağ olun`, `təşəkkür`), recommendations (`tövsiyə`, `hamıya`), mobile operators.

**Comparison.** Negative topics describe concrete problems; positive topics are more general (praise, thanks, recommendations). Part of the positive-star reviews actually report problems, which matches the label noise found in Section 7.

Topic tables were saved to files.

## 15. Web demo

**Purpose.** Show that the whole pipeline works on raw text, not only on prepared datasets.

**What was done.** The saved XLM-R model and tokenizer were wrapped into a simple Gradio interface: the user types a review, the same `clean()` function is applied, and the model returns the probability of each class. The interface was launched.

## 16. Conclusions

**How the sentiment labels were defined.** Star ratings were mapped to three classes (1-2 negative, 3 neutral, 4-5 positive). Contradictions between stars and text were inspected with random samples, word frequencies and a keyword filter. They exist in both directions (praise with low stars, problems with high stars), but the labels were kept as the users' own ratings.

**How noise and imbalance were handled.** Noise: light cleaning that keeps emojis, case and common words; removal of empty and meaningless reviews; duplicates removed in train only; test overlap measured and reported separately. Imbalance: only the training part was balanced, validation and test were kept natural, and macro-F1 with per-class metrics was used instead of accuracy.

**How the models performed.** XLM-R and the TF-IDF baseline perform almost the same (test macro-F1 0.580 vs 0.578; on new reviews 0.565 vs 0.554). Both handle negative and positive reviews reasonably well and are weak on neutral. Many of the model's most confident "errors" are label errors, so the test scores probably understate the quality.

**What the topics reveal about customer feedback.** Complaints are concrete and mostly about access (login, registration, codes), payments and card operations, app stability (freezing, not loading), taxi pricing and routes, and advertising messages. Praise is general (the app is good, thanks, recommendations, ease of use). Some high-star reviews are really questions or problem reports, and some reviews suggest improvements.

## 17. Output files

- Notebook with all stages in order.
- Evaluation pack: `eval_test_report.csv` (per-class metrics on test), `eval_confusion_matrix_test.png`, `eval_comparison.csv` (baseline vs XLM-R on the full and new test reviews), `topics_negative.csv`, `topics_positive.csv` (topic sizes, shares and keywords).
- Intermediate data: `train_clean.csv`, `test_clean.csv`, `train_labeled.csv`, `test_labeled.csv`, `train_bal.csv`, `val.csv`.
- Saved model and tokenizer: `xlmr_sentiment/`.
