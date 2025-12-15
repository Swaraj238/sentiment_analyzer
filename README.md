# Sentiment Analysis (TF-IDF + SVM)

An end-to-end sentiment classifier for Amazon Kindle reviews with a strong, simple baseline (TF-IDF + Linear SVM), plus comparative baselines (Bag-of-Words + NB, Word2Vec + RF), rich EDA, error analysis, and model interpretability.

## Dataset
- Source: Amazon Kindle reviews (CSV in `Dataset/`)
- Labels: Binary sentiment from ratings (>=3 → POSITIVE, else NEGATIVE)

## Approach
1. Preprocessing: URL removal, lowercasing, tokenization, alphabetic filtering, stopword removal, lemmatization.
2. Features: TF-IDF with n-grams.
3. Model: LinearSVC. CV grid-search over `C`, `ngram_range`, `min_df`, `max_features`.
4. Evaluation: Stratified 5-fold CV (F1), held-out test set metrics, confusion matrix, error slices.
5. Interpretability: Top positive/negative n-grams; hardest FP/FN examples by margin.
6. Extras: Bag-of-Words + NB, TF-IDF + NB, Word2Vec + RandomForest.

## Quickstart
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install deps
pip install -r requirements.txt

# Launch Jupyter (or open in VS Code)
pip install jupyter
jupyter notebook executed.ipynb
```

## CLI Training (Optional)
```bash
# Train TF-IDF + LinearSVC and save artifacts
python src/train.py \
  --data Dataset/all_kindle_review.csv \
  --output artifacts \
  --ngrams 1,2 \
  --max-features 30000 \
  --min-df 2 \
  --C 1.0
```

## Reproduce Results (Notebook)
- Open `executed.ipynb` and run cells top-to-bottom:
  - Download NLTK assets
  - Load/clean data, EDA
  - Train TF-IDF + SVM baseline
  - Run CV tuning block (Section 4.6)
  - Review metrics, error analysis, and top n-grams
  - Save artifacts to `artifacts/`

## Assumptions & Trade-offs
- Ratings >=3 labeled POSITIVE (reasonable for Amazon reviews)
- TF-IDF (1–2 grams) capped features (20–30k) for speed
- Linear SVM chosen for strong baseline + efficiency

## Next Steps
- Calibrated probabilities (`CalibratedClassifierCV`) for thresholding
- Domain lexicons or sentiment priors
- Transformer baseline (DistilBERT) for comparison
