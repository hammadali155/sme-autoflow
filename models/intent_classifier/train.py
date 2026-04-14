"""
train.py -- Train the Intent Classifier model.

Pipeline: TF-IDF vectoriser -> Logistic Regression
Input:    Free-text business description
Output:   One of the intent categories (e.g. email_automation, data_sync, ...)

Saves:
    intent_model.pkl   -- the full sklearn Pipeline
    label_encoder.pkl  -- LabelEncoder for mapping labels <-> indices

Usage:
    python models/intent_classifier/train.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_CSV: Path = DATA_DIR / "labeled_dataset.csv"

MODEL_PATH: Path = SCRIPT_DIR / "intent_model.pkl"
ENCODER_PATH: Path = SCRIPT_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
LR_MAX_ITER: int = 1000
MIN_CLASS_SIZE: int = 5  # classes smaller than this get merged into 'general'


def load_data() -> pd.DataFrame:
    """Load and validate the labelled dataset.

    Returns:
        A DataFrame with columns 'description' and 'intent'.
    """
    if not DATASET_CSV.exists():
        print(f"[ERROR] Dataset not found at {DATASET_CSV}")
        print("       Run  python data/build_dataset.py  first.")
        sys.exit(1)

    df: pd.DataFrame = pd.read_csv(DATASET_CSV)

    # Drop rows with missing descriptions
    df = df.dropna(subset=["description"])
    df = df[df["description"].str.strip().astype(bool)]

    # Merge tiny classes (< MIN_CLASS_SIZE samples) into 'general'
    # to avoid stratified-split errors and improve model stability.
    counts = df["intent"].value_counts()
    small_classes = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if small_classes:
        df["intent"] = df["intent"].replace(
            {cls: "general" for cls in small_classes},
        )
        print(f"[MERGE] Merged small classes into 'general': {small_classes}")

    print(f"[DATA] Loaded {len(df)} rows from {DATASET_CSV.name}")
    print(f"       Classes: {df['intent'].nunique()}  ->  "
          f"{sorted(df['intent'].unique())}\n")
    return df


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train the intent classifier pipeline and save artefacts.

    Args:
        df: DataFrame with 'description' and 'intent' columns.

    Returns:
        A dict of evaluation metrics.
    """
    # --- Encode labels ---
    le = LabelEncoder()
    y: np.ndarray = le.fit_transform(df["intent"])
    X: pd.Series = df["description"]

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"[SPLIT] Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # --- Pipeline ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=LR_MAX_ITER,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred: np.ndarray = pipeline.predict(X_test)

    # --- Metrics ---
    acc: float = accuracy_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred, average="weighted")
    prec: float = precision_score(y_test, y_pred, average="weighted")
    rec: float = recall_score(y_test, y_pred, average="weighted")

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
    }

    print("--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:>20s}: {v}")

    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred, target_names=le.classes_,
    ))

    # --- Save ---
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"[SAVED] {MODEL_PATH.name}   ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"[SAVED] {ENCODER_PATH.name}  ({ENCODER_PATH.stat().st_size:,} bytes)")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: load data, train, evaluate, save."""
    print("=" * 60)
    print("  Intent Classifier -- Training")
    print("=" * 60 + "\n")

    df = load_data()
    metrics = train_model(df)

    print("\n[DONE] Intent classifier training complete.")


if __name__ == "__main__":
    main()
