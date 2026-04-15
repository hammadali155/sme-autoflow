"""
train.py -- Intent Classifier Training Script
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Pipeline: TF-IDF + Logistic Regression (multi-class)
Input:    data/labeled_dataset.csv  (description, nodes, intent)
Output:   intent_model.pkl, label_encoder.pkl, confusion_matrix.png

Usage:
    python models/intent_classifier/train.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")  # non-interactive backend safe for scripts

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent
DATASET_CSV: Path = PROJECT_ROOT / "data" / "labeled_dataset.csv"
MODEL_PATH: Path = SCRIPT_DIR / "intent_model.pkl"
ENCODER_PATH: Path = SCRIPT_DIR / "label_encoder.pkl"
CM_PNG: Path = SCRIPT_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_CLASS_SIZE: int = 5
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load labeled_dataset.csv and validate it exists."""
    if not DATASET_CSV.exists():
        print(f"[ERROR] Dataset not found: {DATASET_CSV}")
        print("       Run  python data/build_dataset.py  first.")
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)
    print(f"[1/9] Loaded {len(df)} rows from {DATASET_CSV.name}")
    return df


# ---------------------------------------------------------------------------
# 2. Drop null descriptions / intents
# ---------------------------------------------------------------------------
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where description or intent is null or blank."""
    before = len(df)
    df = df.dropna(subset=["description", "intent"])
    df = df[df["description"].str.strip().astype(bool)]
    df = df[df["intent"].str.strip().astype(bool)]
    dropped = before - len(df)
    print(f"[2/9] Dropped {dropped} null/blank rows  ->  {len(df)} remaining")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Filter small classes
# ---------------------------------------------------------------------------
def filter_small_classes(df: pd.DataFrame, min_size: int = MIN_CLASS_SIZE) -> pd.DataFrame:
    """Remove intent categories with fewer than min_size samples."""
    counts = df["intent"].value_counts()
    keep = counts[counts >= min_size].index
    removed_classes = sorted(set(df["intent"].unique()) - set(keep))
    df = df[df["intent"].isin(keep)].reset_index(drop=True)
    print(
        f"[3/9] Filtered classes with < {min_size} samples. "
        f"Removed: {removed_classes}  ->  {df['intent'].nunique()} classes remain"
    )
    return df


# ---------------------------------------------------------------------------
# 4. Encode labels
# ---------------------------------------------------------------------------
def encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode intent strings to integer indices."""
    le = LabelEncoder()
    y = le.fit_transform(df["intent"])
    print(f"[4/9] Encoded {len(le.classes_)} classes: {list(le.classes_)}")
    return y, le


# ---------------------------------------------------------------------------
# 5. Train / test split
# ---------------------------------------------------------------------------
def split_data(
    X: pd.Series, y: np.ndarray
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[5/9] Split -> Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6 & 7. Build pipeline and fit
# ---------------------------------------------------------------------------
def build_and_train(
    X_train: pd.Series, y_train: np.ndarray
) -> Pipeline:
    """Build TF-IDF + LR pipeline and fit on training data."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    print("[6/9] Pipeline built: TfidfVectorizer(max_features=5000, ngram_range=(1,2)) + LogisticRegression(C=1.0)")
    pipeline.fit(X_train, y_train)
    print("[7/9] Training complete")
    return pipeline


# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------
def evaluate(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> None:
    """Print accuracy, classification report, and save confusion matrix."""
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[8/9] Evaluation on test set ({len(y_test)} samples):")
    print(f"      Accuracy: {acc:.4f}  ({acc*100:.2f}%)\n")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Intent Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Confusion matrix saved to {CM_PNG.name}")


# ---------------------------------------------------------------------------
# 9. Save artefacts
# ---------------------------------------------------------------------------
def save_artefacts(pipeline: Pipeline, le: LabelEncoder) -> None:
    """Save the trained pipeline and label encoder with joblib."""
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n[9/9] intent_model.pkl   saved ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"      label_encoder.pkl   saved ({ENCODER_PATH.stat().st_size:,} bytes)")
    print("\nModel saved successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry-point: load -> clean -> encode -> split -> train -> evaluate -> save."""
    print("=" * 60)
    print("  Intent Classifier — Training")
    print("=" * 60 + "\n")

    df = load_data()
    df = drop_nulls(df)
    df = filter_small_classes(df)

    y, le = encode_labels(df)
    X = df["description"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_and_train(X_train, y_train)
    evaluate(pipeline, X_test, y_test, le)
    save_artefacts(pipeline, le)


if __name__ == "__main__":
    main()
