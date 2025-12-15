import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None


def ensure_nltk():
    if nltk is None:
        return
    for r in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{r}")
        except LookupError:
            nltk.download(r, quiet=True)


def clean_text_series(texts: pd.Series, use_clean: bool = True, remove_stop: bool = True, lemmatize: bool = True) -> pd.Series:
    if not use_clean:
        return texts.astype(str).str.lower()

    ensure_nltk()
    sw = set(stopwords.words("english")) if nltk else set()
    lem = WordNetLemmatizer() if nltk else None
    url_re = re.compile(r"http\S+|www\S+")
    alpha_re = re.compile(r"^[a-zA-Z]+$")

    def _one(x: str) -> str:
        x = x if isinstance(x, str) else ""
        x = url_re.sub("", x)
        toks = word_tokenize(x.lower()) if nltk else re.findall(r"[a-zA-Z]+", x.lower())
        toks = [t for t in toks if alpha_re.match(t)]
        if remove_stop and sw:
            toks = [t for t in toks if t not in sw]
        if lemmatize and lem:
            toks = [lem.lemmatize(t) for t in toks]
        return " ".join(toks)

    return texts.apply(_one)


def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + LinearSVC sentiment model")
    ap.add_argument("--data", required=True, help="Path to CSV with columns: rating, reviewText")
    ap.add_argument("--output", default="artifacts", help="Directory to save artifacts")
    ap.add_argument("--ngrams", default="1,2", help="n-gram range, e.g. 1,2 or 1,1")
    ap.add_argument("--max-features", type=int, default=30000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found at {data_path.resolve()}")

    df = pd.read_csv(data_path)
    if not {"rating", "reviewText"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: rating, reviewText")

    # Binary labels
    y = (df["rating"] >= 3).astype(int)

    # Clean text
    X_clean = clean_text_series(df["reviewText"], use_clean=True, remove_stop=True, lemmatize=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Vectorize
    n_lo, n_hi = map(int, args.ngrams.split(","))
    tfidf = TfidfVectorizer(ngram_range=(n_lo, n_hi), max_features=args.max_features, min_df=args.min_df)
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v = tfidf.transform(X_test)

    # Train
    clf = LinearSVC(C=args.C)
    clf.fit(X_train_v, y_train)

    # Evaluate
    pred = clf.predict(X_test_v)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)

    print("\nTest Metrics (TF-IDF + LinearSVC)")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(tfidf, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf, out_dir / "svm_tfidf.joblib")
    print(f"Saved artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
