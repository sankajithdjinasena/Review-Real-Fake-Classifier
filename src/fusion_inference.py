from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from feature_engineering import ENGINEERED_COLS, extract_features_batch

REQUIRED_FILES = [
    "tfidf_vectorizer.joblib",
    "scaler.joblib",
    "logistic_regression.joblib",
    "linear_svm.joblib",
    "metadata.json",
]


def load_fusion_artifacts(model_dir: str) -> dict | None:
    """Loads everything train_fusion_model.py saved. Returns None if the
    folder doesn't exist or is missing any expected file (mirrors the
    None-on-failure convention used by load_model() for DistilBERT)."""
    import joblib  # imported lazily so importing this module doesn't require it

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None

    missing = [f for f in REQUIRED_FILES if not (model_dir / f).exists()]
    if missing:
        return None

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    return {
        "vectorizer": joblib.load(model_dir / "tfidf_vectorizer.joblib"),
        "scaler": joblib.load(model_dir / "scaler.joblib"),
        "logreg": joblib.load(model_dir / "logistic_regression.joblib"),
        "svm": joblib.load(model_dir / "linear_svm.joblib"),
        "classes": metadata["classes"],  # e.g. ['fake', 'real'], alphabetical
        "metadata": metadata,
    }


def _select_model(artifacts: dict, which: str):
    if which not in ("logreg", "svm"):
        raise ValueError(f"which must be 'logreg' or 'svm', got {which!r}")
    return artifacts["logreg"] if which == "logreg" else artifacts["svm"]


def _build_feature_vector(text: str, artifacts: dict):
    """Returns (fused sparse feature vector, engineered feature values array)."""
    x_tfidf = artifacts["vectorizer"].transform([text])
    eng_df = extract_features_batch([text])
    x_eng = artifacts["scaler"].transform(eng_df[ENGINEERED_COLS])
    x_fused = hstack([x_tfidf, csr_matrix(x_eng)]).tocsr()
    return x_fused, x_tfidf, x_eng


def predict_fusion(text: str, artifacts: dict, which: str = "logreg") -> dict:
    """Same return shape as app.py's predict() for DistilBERT, so both
    branches of the UI can share identical rendering code."""
    model = _select_model(artifacts, which)
    x_fused, _, _ = _build_feature_vector(text, artifacts)

    proba = model.predict_proba(x_fused)[0]
    classes = artifacts["classes"]
    fake_idx = classes.index("fake")
    real_idx = classes.index("real")

    prob_fake = float(proba[fake_idx])
    prob_real = float(proba[real_idx])
    label = "fake" if prob_fake >= prob_real else "real"

    return {
        "label": label,
        "confidence": max(prob_fake, prob_real),
        "prob_real": prob_real,
        "prob_fake": prob_fake,
    }


def _get_linear_coefficients(model) -> np.ndarray:
    """Returns a 1-D array of coefficients over the fused feature space,
    oriented toward classes_[1] (sklearn's convention for binary coef_)."""
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()

    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for cc in model.calibrated_classifiers_:
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                coefs.append(np.asarray(base.coef_).ravel())
        if coefs:
            return np.mean(coefs, axis=0)

    raise ValueError("Could not extract linear coefficients from this model.")


def explain_fusion(text: str, artifacts: dict, which: str = "logreg") -> pd.DataFrame:
    model = _select_model(artifacts, which)
    classes = artifacts["classes"]

    x_fused, x_tfidf, x_eng = _build_feature_vector(text, artifacts)
    coef = _get_linear_coefficients(model)

    # sklearn's binary coef_ points toward classes_[1]; flip sign if that's
    # not "fake", so positive always means "toward fake" like the DistilBERT branch.
    sign = 1.0 if classes[1] == "fake" else -1.0
    coef = coef * sign

    tfidf_names = artifacts["vectorizer"].get_feature_names_out()
    all_names = list(tfidf_names) + list(ENGINEERED_COLS)
    n_tfidf = len(tfidf_names)

    contributions: dict[str, float] = {}

    x_tfidf_coo = x_tfidf.tocoo()
    for col_idx, value in zip(x_tfidf_coo.col, x_tfidf_coo.data):
        name = all_names[col_idx]
        contributions[name] = contributions.get(name, 0.0) + float(coef[col_idx] * value)

    for j, value in enumerate(x_eng[0]):
        col_idx = n_tfidf + j
        name = all_names[col_idx]
        contributions[name] = contributions.get(name, 0.0) + float(coef[col_idx] * value)

    rows = [{"display_token": k, "shap_value": v} for k, v in contributions.items() if v != 0.0]
    return pd.DataFrame(rows, columns=["display_token", "shap_value"])
