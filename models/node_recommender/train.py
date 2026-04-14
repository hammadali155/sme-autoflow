"""
train.py -- Train the Node Recommender model (multi-label).

Pipeline: TF-IDF vectoriser -> OneVsRest(Logistic Regression)
Input:    Free-text business description
Output:   List of predicted n8n node types

Saves:
    node_model.pkl -- the full sklearn Pipeline
    mlb.pkl        -- MultiLabelBinarizer for encoding/decoding node labels

Usage:
    python models/node_recommender/train.py
"""

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
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_CSV: Path = DATA_DIR / "labeled_dataset.csv"

MODEL_PATH: Path = SCRIPT_DIR / "node_model.pkl"
MLB_PATH: Path = SCRIPT_DIR / "mlb.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
TFIDF_MAX_FEATURES: int = 5000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
LR_MAX_ITER: int = 1000
MIN_NODE_FREQUENCY: int = 5  # drop very rare nodes to keep model stable


def load_data() -> pd.DataFrame:
    """Load the labelled dataset and parse the comma-separated nodes column.

    Returns:
        DataFrame with 'description' and 'nodes' (list[str]) columns.
    """
    if not DATASET_CSV.exists():
        print(f"[ERROR] Dataset not found at {DATASET_CSV}")
        print("       Run  python data/build_dataset.py  first.")
        sys.exit(1)

    df: pd.DataFrame = pd.read_csv(DATASET_CSV)

    # Drop rows with missing descriptions or nodes
    df = df.dropna(subset=["description", "nodes"])
    df = df[df["description"].str.strip().astype(bool)]
    df = df[df["nodes"].str.strip().astype(bool)]

    # Parse comma-separated node strings into lists
    df["nodes"] = df["nodes"].apply(lambda x: [n.strip() for n in str(x).split(",") if n.strip()])

    # Filter out rows with empty node lists
    df = df[df["nodes"].apply(len) > 0].reset_index(drop=True)

    print(f"[DATA] Loaded {len(df)} rows from {DATASET_CSV.name}")

    # Count unique nodes
    all_nodes = [n for nodes in df["nodes"] for n in nodes]
    unique_nodes = set(all_nodes)
    print(f"       Unique node types: {len(unique_nodes)}\n")

    return df


def filter_rare_nodes(
    df: pd.DataFrame, min_freq: int = MIN_NODE_FREQUENCY,
) -> pd.DataFrame:
    """Remove node types that appear fewer than min_freq times.

    Args:
        df: DataFrame with 'nodes' column (list[str]).
        min_freq: Minimum occurrence count to keep a node type.

    Returns:
        DataFrame with rare nodes stripped from each row.
    """
    # Count frequencies
    from collections import Counter
    node_counts: Counter = Counter(n for nodes in df["nodes"] for n in nodes)
    keep: set[str] = {n for n, c in node_counts.items() if c >= min_freq}
    removed: int = len(node_counts) - len(keep)

    # Filter
    df["nodes"] = df["nodes"].apply(lambda ns: [n for n in ns if n in keep])
    df = df[df["nodes"].apply(len) > 0].reset_index(drop=True)

    print(f"[FILTER] Kept {len(keep)} node types (removed {removed} with freq < {min_freq})")
    print(f"         Remaining rows: {len(df)}\n")
    return df


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """Train the multi-label node recommender pipeline and save artefacts.

    Args:
        df: DataFrame with 'description' and 'nodes' columns.

    Returns:
        A dict of evaluation metrics.
    """
    # --- Binarise labels ---
    mlb = MultiLabelBinarizer()
    y: np.ndarray = mlb.fit_transform(df["nodes"])
    X: pd.Series = df["description"]

    print(f"[LABELS] {len(mlb.classes_)} node classes after binarisation")

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"[SPLIT]  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # --- Pipeline ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=LR_MAX_ITER,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred: np.ndarray = pipeline.predict(X_test)

    # --- Metrics ---
    h_loss: float = hamming_loss(y_test, y_pred)
    acc: float = accuracy_score(y_test, y_pred)  # exact-match ratio
    f1_micro: float = f1_score(y_test, y_pred, average="micro")
    f1_macro: float = f1_score(y_test, y_pred, average="macro")
    prec: float = precision_score(y_test, y_pred, average="micro")
    rec: float = recall_score(y_test, y_pred, average="micro")

    metrics: dict[str, Any] = {
        "hamming_loss": round(h_loss, 4),
        "exact_match_accuracy": round(acc, 4),
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_micro": round(prec, 4),
        "recall_micro": round(rec, 4),
    }

    print("--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:>25s}: {v}")

    # --- Save ---
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    print(f"\n[SAVED] {MODEL_PATH.name}  ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"[SAVED] {MLB_PATH.name}       ({MLB_PATH.stat().st_size:,} bytes)")

    # --- Sample predictions ---
    print("\n--- Sample Predictions (first 5 test rows) ---")
    for i in range(min(5, len(X_test))):
        idx = X_test.index[i]
        desc_preview = X_test.iloc[i][:80].encode("ascii", errors="replace").decode() + "..."
        true_nodes = mlb.inverse_transform(y_test[i:i+1])[0]
        pred_nodes = mlb.inverse_transform(y_pred[i:i+1])[0]
        print(f"\n  [{i+1}] {desc_preview}")
        print(f"      True:  {list(true_nodes)}")
        print(f"      Pred:  {list(pred_nodes)}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: load data, filter, train, evaluate, save."""
    print("=" * 60)
    print("  Node Recommender -- Training (Multi-Label)")
    print("=" * 60 + "\n")

    df = load_data()
    df = filter_rare_nodes(df)
    metrics = train_model(df)

    print("\n[DONE] Node recommender training complete.")


if __name__ == "__main__":
    main()
