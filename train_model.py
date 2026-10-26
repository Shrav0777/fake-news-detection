"""
train_model.py — Dataset Loading, Preprocessing, Training & Model Selection
=============================================================================
This script performs the complete machine-learning pipeline:

1.  Load the WELFake dataset from data/WELFake_Dataset.csv.
2.  Inspect and clean the data (missing values, duplicates, encoding).
3.  Combine title + text into a single feature column.
4.  Split into training (80%) and testing (20%) sets — stratified.
5.  Fit TF-IDF vectorizer on training data only (no data leakage).
6.  Train three classifiers:
        a) Logistic Regression
        b) Multinomial Naive Bayes
        c) Passive Aggressive Classifier
7.  Evaluate each on the test set.
8.  Select the best model based on F1-score (weighted).
9.  Save the best model, vectorizer, and evaluation results to disk.

Usage:
    python train_model.py

Output artifacts (saved to models/ and results/):
    - models/tfidf_vectorizer.joblib
    - models/best_model.joblib
    - models/model_metadata.json
    - results/evaluation_metrics.json
    - results/class_distribution.png
    - results/model_comparison_accuracy.png
    - results/model_comparison_metrics.png
    - results/confusion_matrix.png
    - results/roc_curve.png   (where supported)
"""

import os
import sys
import json
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# We use the exact same clean_text function from predict.py to guarantee
# consistency between training and inference.
from predict import clean_text, combine_headline_body

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")

# Create output directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1 — DATA LOADING & INSPECTION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the WELFake CSV dataset.

    The WELFake dataset has columns:
        Unnamed: 0  — serial number (dropped)
        title       — news headline
        text        — news body
        label       — 0 = Fake, 1 = Real
    """
    if not os.path.isfile(path):
        logger.error("Dataset not found at: %s", path)
        logger.error(
            "Please download the WELFake dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            "Extract the ZIP and place WELFake_Dataset.csv in the data/ directory."
        )
        sys.exit(1)

    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)
    logger.info("Raw dataset shape: %s", df.shape)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the raw dataset."""
    print("\n" + "=" * 70)
    print("  DATASET INSPECTION")
    print("=" * 70)
    print(f"\n  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns          : {list(df.columns)}")
    print(f"\n  Data types:\n{df.dtypes.to_string()}")
    print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
    print(f"\n  Duplicates       : {df.duplicated().sum():,}")
    if "label" in df.columns:
        print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
    print("=" * 70 + "\n")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 2 — DATA CLEANING & PREPROCESSING                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the dataset for training.

    Steps:
        1. Drop the unnamed index column if present.
        2. Ensure 'label' column exists and contains only 0/1.
        3. Convert title and text to string; fill NaN with empty string.
        4. Combine title + text into a single 'content' column.
        5. Apply text cleaning.
        6. Drop rows where cleaned content is empty.
        7. Remove exact duplicates on content + label.
    """
    logger.info("Starting data preprocessing...")

    # 1. Drop serial-number column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        logger.info("Dropped 'Unnamed: 0' column.")

    # 2. Validate labels
    if "label" not in df.columns:
        logger.error("Column 'label' not found. Available: %s", list(df.columns))
        sys.exit(1)

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    invalid_labels = df["label"].isnull().sum()
    if invalid_labels > 0:
        logger.warning("Dropping %d rows with invalid labels.", invalid_labels)
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Ensure binary
    valid = df["label"].isin([0, 1])
    if not valid.all():
        n_bad = (~valid).sum()
        logger.warning("Dropping %d rows with labels not in {0, 1}.", n_bad)
        df = df[valid]

    # 3. Handle NaN in text columns
    for col in ["title", "text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # 4. Combine title and text
    df["content"] = df.apply(
        lambda row: combine_headline_body(row["title"], row["text"]), axis=1
    )

    # 5. Clean text
    logger.info("Cleaning text (this may take a minute)...")
    df["content_clean"] = df["content"].apply(clean_text)

    # 6. Drop empty rows
    empty_mask = df["content_clean"].str.strip() == ""
    n_empty = empty_mask.sum()
    if n_empty > 0:
        logger.warning("Dropping %d rows with empty cleaned text.", n_empty)
        df = df[~empty_mask]

    # 7. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["content_clean", "label"])
    after = len(df)
    if before != after:
        logger.info("Removed %d duplicate rows.", before - after)

    df = df.reset_index(drop=True)
    logger.info("Preprocessed dataset shape: %s", df.shape)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 3 — VISUALIZATION HELPERS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Consistent color palette
COLOR_REAL = "#2ecc71"   # green
COLOR_FAKE = "#e74c3c"   # red
PALETTE = [COLOR_FAKE, COLOR_REAL]
MODEL_COLORS = ["#3498db", "#e67e22", "#9b59b6"]  # blue, orange, purple


def plot_class_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart showing REAL vs. FAKE article counts."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label"].value_counts().sort_index()
    labels = ["FAKE (0)", "REAL (1)"]
    bars = ax.bar(labels, counts.values, color=PALETTE, edgecolor="white", width=0.55)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_title("Class Distribution — FAKE vs REAL", fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of Articles", fontsize=12)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved class distribution plot → %s", save_path)


def plot_model_comparison_accuracy(results: dict, save_path: str) -> None:
    """Bar chart comparing accuracy of all models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    bars = ax.bar(names, accuracies, color=MODEL_COLORS[: len(names)],
                  edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{acc:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylim(min(accuracies) - 0.05, 1.0)
    ax.set_title("Model Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved accuracy comparison → %s", save_path)


def plot_model_comparison_metrics(results: dict, save_path: str) -> None:
    """Grouped bar chart for precision, recall, and F1 across models."""
    metrics = ["precision", "recall", "f1_score"]
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    metric_colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=m.replace("_", " ").title(),
                      color=metric_colors[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(min(min(results[n][m] for m in metrics) for n in names) - 0.05, 1.0)
    ax.set_title("Precision / Recall / F1-Score Comparison", fontsize=15, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved metrics comparison → %s", save_path)


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str) -> None:
    """Heatmap of the confusion matrix for the best model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
        linewidths=1, linecolor="white", ax=ax,
        annot_kws={"size": 15, "fontweight": "bold"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", save_path)


def plot_roc_curves(roc_data: dict, save_path: str) -> None:
    """Plot ROC curves for all models that support probability scores."""
    if not roc_data:
        logger.info("No ROC data available — skipping ROC plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (name, data) in enumerate(roc_data.items()):
        ax.plot(data["fpr"], data["tpr"],
                label=f'{name} (AUC = {data["auc"]:.4f})',
                color=MODEL_COLORS[idx % len(MODEL_COLORS)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curves → %s", save_path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 4 — TRAINING PIPELINE                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_tfidf(X_train: pd.Series, max_features: int = 50_000) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the training data only.

    TF-IDF (Term Frequency–Inverse Document Frequency) explained simply:
    ────────────────────────────────────────────────────────────────────
    Imagine you have 1,000 news articles and the word "president" appears
    in 100 of them but "scandal" appears in only 5.  TF-IDF gives a higher
    weight to "scandal" because it is rarer and therefore more informative
    for distinguishing between articles.

    • TF  = how often a word appears in ONE document.
    • IDF = how rare the word is across ALL documents.
    • TF-IDF = TF × IDF  → common words get low scores, rare informative
      words get high scores.

    Example:
        Document: "the cat sat on the mat"
        TF("the") = 2/6   → common → low IDF → low TF-IDF
        TF("cat") = 1/6   → rarer  → high IDF → higher TF-IDF
    """
    logger.info(
        "Fitting TF-IDF vectorizer (max_features=%d) on %d training samples...",
        max_features, len(X_train),
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),       # unigrams + bigrams
        min_df=3,                 # ignore very rare terms
        max_df=0.95,              # ignore terms appearing in >95% of docs
        sublinear_tf=True,        # apply log normalization to TF
    )
    vectorizer.fit(X_train)
    logger.info("Vocabulary size: %d terms", len(vectorizer.vocabulary_))
    return vectorizer


def train_and_evaluate(
    X_train_tfidf,
    X_test_tfidf,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict, dict]:
    """
    Train three classifiers and evaluate each on the test set.

    Returns
    -------
    all_results : dict   – metrics per model
    best_name   : str    – name of the best model (by weighted F1)
    roc_data    : dict   – FPR/TPR/AUC per model (where available)
    predictions : dict   – y_pred per model
    """
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Multinomial NB": MultinomialNB(alpha=0.1),
        "Passive Aggressive": PassiveAggressiveClassifier(
            max_iter=1000, C=1.0, random_state=42
        ),
    }

    all_results = {}
    roc_data = {}
    predictions = {}

    for name, clf in classifiers.items():
        logger.info("Training %s...", name)
        t0 = time.time()
        clf.fit(X_train_tfidf, y_train)
        train_time = time.time() - t0

        y_pred = clf.predict(X_test_tfidf)
        predictions[name] = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(
            y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True
        )

        all_results[name] = {
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1_score": round(f1, 5),
            "train_time_sec": round(train_time, 2),
            "classification_report": report,
        }

        logger.info(
            "  %s → Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  (%.1fs)",
            name, acc, prec, rec, f1, train_time,
        )

        # ROC / AUC — requires probability or decision-function scores
        try:
            if hasattr(clf, "predict_proba"):
                y_scores = clf.predict_proba(X_test_tfidf)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_scores = clf.decision_function(X_test_tfidf)
            else:
                y_scores = None

            if y_scores is not None:
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                auc_val = roc_auc_score(y_test, y_scores)
                roc_data[name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": round(auc_val, 5),
                }
                all_results[name]["roc_auc"] = round(auc_val, 5)
                logger.info("  %s → ROC-AUC=%.4f", name, auc_val)
        except Exception as e:
            logger.warning("  Could not compute ROC for %s: %s", name, e)

    return all_results, roc_data, predictions, classifiers


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 5 — MODEL SELECTION & PERSISTENCE                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def select_best_model(all_results: dict, classifiers: dict):
    """Select the model with the highest weighted F1-score."""
    best_name = max(all_results, key=lambda n: all_results[n]["f1_score"])
    best_clf = classifiers[best_name]
    logger.info(
        "Best model: %s (F1=%.4f)",
        best_name, all_results[best_name]["f1_score"],
    )
    return best_name, best_clf


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("\n" + "═" * 70)
    print("  FAKE NEWS DETECTION — MODEL TRAINING PIPELINE")
    print("═" * 70 + "\n")

    # ── 1. Load ──────────────────────────────────────────────────────────
    df = load_dataset(DATASET_PATH)
    inspect_dataset(df)

    # ── 2. Preprocess ────────────────────────────────────────────────────
    df = preprocess_dataset(df)
    print(f"\n  After preprocessing: {len(df):,} rows")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # Save dataset stats
    dataset_info = {
        "total_samples": len(df),
        "fake_count": int((df["label"] == 0).sum()),
        "real_count": int((df["label"] == 1).sum()),
        "columns": list(df.columns),
        "sample_rows": df[["title", "label"]].head(5).to_dict(orient="records"),
    }

    # ── 3. Plot class distribution ───────────────────────────────────────
    plot_class_distribution(df, os.path.join(RESULTS_DIR, "class_distribution.png"))

    # ── 4. Train-test split ──────────────────────────────────────────────
    X = df["content_clean"]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info("Train set: %d | Test set: %d", len(X_train), len(X_test))

    # ── 5. TF-IDF ────────────────────────────────────────────────────────
    vectorizer = build_tfidf(X_train)
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    logger.info("TF-IDF matrix: train=%s  test=%s", X_train_tfidf.shape, X_test_tfidf.shape)

    # ── 6. Train and evaluate ────────────────────────────────────────────
    all_results, roc_data, predictions, classifiers = train_and_evaluate(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # ── 7. Select best model ─────────────────────────────────────────────
    best_name, best_clf = select_best_model(all_results, classifiers)
    best_pred = predictions[best_name]

    # ── 8. Generate plots ────────────────────────────────────────────────
    plot_model_comparison_accuracy(
        all_results, os.path.join(RESULTS_DIR, "model_comparison_accuracy.png")
    )
    plot_model_comparison_metrics(
        all_results, os.path.join(RESULTS_DIR, "model_comparison_metrics.png")
    )
    plot_confusion_matrix(
        y_test, best_pred, best_name,
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )
    plot_roc_curves(roc_data, os.path.join(RESULTS_DIR, "roc_curve.png"))

    # ── 9. Save artifacts ────────────────────────────────────────────────
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    logger.info("Saved TF-IDF vectorizer → models/tfidf_vectorizer.joblib")

    joblib.dump(best_clf, os.path.join(MODELS_DIR, "best_model.joblib"))
    logger.info("Saved best model (%s) → models/best_model.joblib", best_name)

    # Model metadata
    model_meta = {
        "best_model_name": best_name,
        "best_model_type": type(best_clf).__name__,
        "tfidf_max_features": vectorizer.max_features,
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(model_meta, f, indent=2)

    # Evaluation results
    eval_output = {
        "dataset_info": dataset_info,
        "model_results": all_results,
        "best_model": best_name,
        "roc_data": {k: {"auc": v["auc"]} for k, v in roc_data.items()},
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_output, f, indent=2)
    logger.info("Saved evaluation metrics → results/evaluation_metrics.json")

    # ── 10. Final summary ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("═" * 70)
    for name, metrics in all_results.items():
        marker = " ★ BEST" if name == best_name else ""
        print(
            f"  {name:25s}  Acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_score']:.4f}  "
            f"Time={metrics['train_time_sec']:.1f}s{marker}"
        )
    print()
    print("  Saved artifacts:")
    print(f"    models/tfidf_vectorizer.joblib")
    print(f"    models/best_model.joblib")
    print(f"    models/model_metadata.json")
    print(f"    results/evaluation_metrics.json")
    print(f"    results/*.png  (5 plots)")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
