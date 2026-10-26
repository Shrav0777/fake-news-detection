"""
predict.py — Reusable Prediction Module for Fake News Detection
================================================================
Loads the trained TF-IDF vectorizer and classifier from disk and provides
functions to preprocess new text and predict whether it is REAL or FAKE.

This module is imported by app.py (Streamlit interface) and can also be
used standalone for quick command-line predictions.
"""

import os
import re
import sys
import logging
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "results", "evaluation_metrics.json")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text preprocessing — must match training pipeline exactly
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean a single string of news text.

    Steps:
        1. Convert to lowercase.
        2. Remove URLs.
        3. Remove HTML tags.
        4. Remove non-alphabetic characters (keep spaces).
        5. Collapse multiple spaces.
        6. Strip leading/trailing whitespace.

    This is intentionally simple and mirrors what train_model.py applies so
    that the TF-IDF features are consistent between training and inference.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                       # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                    # non-alpha
    text = re.sub(r"\s+", " ", text).strip()                 # whitespace
    return text


def combine_headline_body(title: str, text: str) -> str:
    """
    Combine a headline and body into a single string for prediction.
    Handles cases where one or both may be empty/None.
    """
    parts = []
    if title and isinstance(title, str) and title.strip():
        parts.append(title.strip())
    if text and isinstance(text, str) and text.strip():
        parts.append(text.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_vectorizer = None
_model = None


def _check_artifacts_exist() -> bool:
    """Return True if both the vectorizer and model files exist on disk."""
    return os.path.isfile(VECTORIZER_PATH) and os.path.isfile(MODEL_PATH)


def load_model(force_reload: bool = False):
    """
    Load the TF-IDF vectorizer and classifier into module-level variables.
    Uses lazy loading — artifacts are loaded once and cached.

    Parameters
    ----------
    force_reload : bool
        If True, reload from disk even if already cached.

    Raises
    ------
    FileNotFoundError
        If model artifacts are missing (training has not been run yet).
    """
    global _vectorizer, _model

    if _vectorizer is not None and _model is not None and not force_reload:
        return  # already loaded

    if not _check_artifacts_exist():
        raise FileNotFoundError(
            "Model artifacts not found. Please run 'python train_model.py' first.\n"
            f"  Expected vectorizer : {VECTORIZER_PATH}\n"
            f"  Expected model      : {MODEL_PATH}"
        )

    logger.info("Loading TF-IDF vectorizer from %s", VECTORIZER_PATH)
    _vectorizer = joblib.load(VECTORIZER_PATH)

    logger.info("Loading classifier from %s", MODEL_PATH)
    _model = joblib.load(MODEL_PATH)

    logger.info("Model artifacts loaded successfully.")


def get_model_info() -> dict:
    """Return basic metadata about the loaded model."""
    load_model()
    return {
        "model_type": type(_model).__name__,
        "vectorizer_type": type(_vectorizer).__name__,
        "n_features": (
            len(_vectorizer.vocabulary_)
            if hasattr(_vectorizer, "vocabulary_")
            else "unknown"
        ),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single(text: str) -> dict:
    """
    Predict whether a single piece of text is REAL or FAKE news.

    Parameters
    ----------
    text : str
        Raw news text (headline, body, or both combined).

    Returns
    -------
    dict with keys:
        label       – "REAL" or "FAKE"
        label_int   – 1 (real) or 0 (fake)
        confidence  – float probability of the predicted class (if available)
        cleaned     – the preprocessed text that was fed into the model
    """
    load_model()

    cleaned = clean_text(text)
    if not cleaned:
        return {
            "label": "UNKNOWN",
            "label_int": -1,
            "confidence": 0.0,
            "cleaned": "",
            "error": "Input text is empty after preprocessing.",
        }

    X = _vectorizer.transform([cleaned])
    prediction = _model.predict(X)[0]

    # Probability / decision function (not all classifiers support predict_proba)
    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        confidence = float(np.max(proba))
    elif hasattr(_model, "decision_function"):
        decision = _model.decision_function(X)[0]
        # Sigmoid-like mapping for display purposes
        confidence = float(1 / (1 + np.exp(-abs(decision))))

    label_str = "REAL" if int(prediction) == 1 else "FAKE"

    return {
        "label": label_str,
        "label_int": int(prediction),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "cleaned": cleaned,
    }


def predict_batch(texts: list[str]) -> list[dict]:
    """
    Predict labels for a list of texts.

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    list of dict – one prediction dict per input text.
    """
    return [predict_single(t) for t in texts]


def predict_dataframe(df: pd.DataFrame, text_column: str = "text",
                      title_column: str | None = None) -> pd.DataFrame:
    """
    Predict for every row of a DataFrame.  Adds 'prediction' and
    'confidence' columns.

    Parameters
    ----------
    df : pd.DataFrame
    text_column : str
        Column containing body text.
    title_column : str or None
        Optional column containing the headline.

    Returns
    -------
    pd.DataFrame – original frame with prediction columns appended.
    """
    results = []
    for _, row in df.iterrows():
        body = str(row.get(text_column, ""))
        title = str(row.get(title_column, "")) if title_column else ""
        combined = combine_headline_body(title, body)
        results.append(predict_single(combined))

    df = df.copy()
    df["prediction"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Quick command-line interface for testing predictions."""
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("=" * 60)
        print("  Fake News Detection — Quick Predictor")
        print("=" * 60)
        text = input("\nEnter news text: ").strip()

    if not text:
        print("Error: No text provided.")
        sys.exit(1)

    try:
        result = predict_single(text)
        print(f"\n  Prediction : {result['label']}")
        if result["confidence"] is not None:
            print(f"  Confidence : {result['confidence']:.2%}")
        print(f"  Cleaned    : {result['cleaned'][:120]}...")
        print()
        print("⚠  Disclaimer: This is a model prediction based on text patterns,")
        print("   not a definitive verification of the truthfulness of this article.")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
