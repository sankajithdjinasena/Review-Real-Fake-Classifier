from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from feature_engineering import ENGINEERED_COLS, extract_features_batch

RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 5000


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="fake")
    rec = recall_score(y_true, y_pred, pos_label="fake")
    f1 = f1_score(y_true, y_pred, pos_label="fake")
    cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (fake):    {prec:.4f}")
    print(f"Recall (fake):       {rec:.4f}")
    print(f"F1 (fake):           {f1:.4f}")
    print("Confusion matrix [rows=actual, cols=predicted] order=[real, fake]:")
    print(cm)
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "model": name, "accuracy": acc, "precision_fake": prec,
        "recall_fake": rec, "f1_fake": f1,
    }


def load_and_prepare(input_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Columns present: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Columns present: {list(df.columns)}")

    missing_engineered = [c for c in ENGINEERED_COLS if c not in df.columns]
    if missing_engineered:
        print(
            f"Engineered columns not found in the input file ({missing_engineered}); "
            f"computing them from '{text_col}' via feature_engineering.py. "
            "This can take a while on large datasets."
        )
        feats = extract_features_batch(df[text_col])
        df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    required = [text_col, label_col] + ENGINEERED_COLS
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after dropping missing values.")

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the training CSV.")
    parser.add_argument("--text-col", default="review_text")
    parser.add_argument("--label-col", default="label_clean")
    parser.add_argument("--output-dir", default="./fusion_model")
    parser.add_argument("--max-tfidf-features", type=int, default=MAX_TFIDF_FEATURES)
    parser.add_argument(
        "--save-augmented-csv",
        default=None,
        help="Optional path to save the CSV with computed engineered columns added, "
        "so you don't have to recompute them next time.",
    )
    args = parser.parse_args()

    df = load_and_prepare(args.input, args.text_col, args.label_col)

    if args.save_augmented_csv:
        df.to_csv(args.save_augmented_csv, index=False)
        print(f"Saved augmented CSV with engineered columns to {args.save_augmented_csv}")

    y = df[args.label_col].values
    idx_train, idx_temp = train_test_split(
        np.arange(len(df)), test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y[idx_temp]
    )

    print(f"Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

    # ---- TF-IDF ----
    vectorizer = TfidfVectorizer(
        max_features=args.max_tfidf_features, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    X_tfidf_train = vectorizer.fit_transform(df[args.text_col].iloc[idx_train].astype(str))
    X_tfidf_test = vectorizer.transform(df[args.text_col].iloc[idx_test].astype(str))

    # ---- Engineered features ----
    scaler = StandardScaler()
    X_eng_train = scaler.fit_transform(df[ENGINEERED_COLS].iloc[idx_train])
    X_eng_test = scaler.transform(df[ENGINEERED_COLS].iloc[idx_test])

    # ---- Fuse ----
    X_train = hstack([X_tfidf_train, csr_matrix(X_eng_train)]).tocsr()
    X_test = hstack([X_tfidf_test, csr_matrix(X_eng_test)]).tocsr()
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"\nFused feature matrix shape: train={X_train.shape}, test={X_test.shape}")
    print(f"  (TF-IDF dims: {X_tfidf_train.shape[1]}, engineered dims: {len(ENGINEERED_COLS)})")

    # ---- Train ----
    results = []

    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train, y_train)
    results.append(evaluate("Fusion (TF-IDF + Engineered) - Logistic Regression", y_test, logreg.predict(X_test)))

    # CalibratedClassifierCV wraps LinearSVC so predict_proba works — a plain
    # LinearSVC only exposes decision_function (a margin, not a probability),
    # and the app needs a real confidence percentage for both models.
    svm = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE), cv=5)
    svm.fit(X_train, y_train)
    results.append(evaluate("Fusion (TF-IDF + Engineered) - Linear SVM (calibrated)", y_test, svm.predict(X_test)))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(pd.DataFrame(results).to_string(index=False))

    # ---- Save artifacts ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(logreg, out_dir / "logistic_regression.joblib")
    joblib.dump(svm, out_dir / "linear_svm.joblib")

    classes = list(logreg.classes_)  # e.g. ['fake', 'real'], alphabetical

    metadata = {
        "engineered_cols": ENGINEERED_COLS,
        "classes": classes,
        "max_tfidf_features": args.max_tfidf_features,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "train_rows": int(len(idx_train)),
        "test_rows": int(len(idx_test)),
        "results": results,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved fusion model artifacts to: {out_dir.resolve()}")
    print("Point the app's 'Fusion model folder path' at this directory.")


if __name__ == "__main__":
    main()
