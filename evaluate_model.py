"""
evaluate_model.py — Standalone Model Evaluation Script
======================================================
Re-evaluates the saved model artifacts against the dataset and regenerates
all evaluation plots and metrics.  Useful when you want to re-run evaluation
without retraining.

Usage:
    python evaluate_model.py

This script:
    1. Loads the saved TF-IDF vectorizer and best model.
    2. Loads and preprocesses the dataset.
    3. Performs the same stratified train/test split (same random_state=42).
    4. Evaluates the model on the test set.
    5. Regenerates all evaluation plots in results/.
    6. Overwrites results/evaluation_metrics.json.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
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

from predict import clean_text, combine_headline_body

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_PATH = os.path.join(DATA_DIR, "WELFake_Dataset.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("\n" + "═" * 70)
    print("  FAKE NEWS DETECTION — MODEL EVALUATION")
    print("═" * 70 + "\n")

    # --- Load artifacts ---
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    for p in [vec_path, model_path]:
        if not os.path.isfile(p):
            logger.error("Missing artifact: %s — run train_model.py first.", p)
            sys.exit(1)

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    model_name = type(model).__name__

    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("best_model_name", model_name)

    logger.info("Loaded model: %s", model_name)

    # --- Load and preprocess dataset ---
    if not os.path.isfile(DATASET_PATH):
        logger.error("Dataset not found: %s", DATASET_PATH)
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])]

    for col in ["title", "text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    df["content"] = df.apply(
        lambda row: combine_headline_body(row["title"], row["text"]), axis=1
    )
    df["content_clean"] = df["content"].apply(clean_text)
    df = df[df["content_clean"].str.strip() != ""]
    df = df.drop_duplicates(subset=["content_clean", "label"]).reset_index(drop=True)

    logger.info("Dataset: %d samples", len(df))

    # --- Reproduce the same split ---
    X = df["content_clean"]
    y = df["label"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True)

    print(f"\n  Model       : {model_name}")
    print(f"  Test size   : {len(y_test):,}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall      : {rec:.4f}")
    print(f"  F1-score    : {f1:.4f}")

    results = {
        model_name: {
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1_score": round(f1, 5),
            "classification_report": report,
        }
    }

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"size": 15, "fontweight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # --- ROC Curve ---
    roc_info = {}
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_tfidf)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test_tfidf)
        else:
            y_scores = None

        if y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc_val = roc_auc_score(y_test, y_scores)
            roc_info[model_name] = {"auc": round(auc_val, 5)}
            results[model_name]["roc_auc"] = round(auc_val, 5)
            print(f"  ROC-AUC     : {auc_val:.4f}")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color="#3498db", linewidth=2,
                    label=f'{model_name} (AUC = {auc_val:.4f})')
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
            ax.legend(fontsize=11, loc="lower right")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150)
            plt.close(fig)
    except Exception as e:
        logger.warning("Could not compute ROC: %s", e)

    # --- Save metrics ---
    dataset_info = {
        "total_samples": len(df),
        "fake_count": int((df["label"] == 0).sum()),
        "real_count": int((df["label"] == 1).sum()),
    }
    eval_output = {
        "dataset_info": dataset_info,
        "model_results": results,
        "best_model": model_name,
        "roc_data": roc_info,
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\n  Results saved to results/")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
