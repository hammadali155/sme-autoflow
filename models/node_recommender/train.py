"""
train.py -- Node Recommender Training Script (Multi-Label)
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Pipeline: TF-IDF + OneVsRest(Logistic Regression)
Input:    data/labeled_dataset.csv  (description, nodes, intent)
Output:   node_model.pkl, mlb.pkl

Usage:
    python models/node_recommender/train.py
"""

import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
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
DATASET_CSV: Path = PROJECT_ROOT / "data" / "labeled_dataset.csv"
MODEL_PATH: Path = SCRIPT_DIR / "node_model.pkl"
MLB_PATH: Path = SCRIPT_DIR / "mlb.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOP_N_NODES: int = 30
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load labeled_dataset.csv."""
    if not DATASET_CSV.exists():
        print(f"[ERROR] Dataset not found: {DATASET_CSV}")
        print("       Run  python data/build_dataset.py  first.")
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)
    print(f"[1/9]  Loaded {len(df)} rows from {DATASET_CSV.name}")
    return df


# ---------------------------------------------------------------------------
# 2. Parse nodes column
# ---------------------------------------------------------------------------
def parse_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Split 'nodes' column (comma-separated string) into lists."""
    df = df.dropna(subset=["nodes", "description"]).copy()
    df["node_list"] = df["nodes"].apply(
        lambda x: [n.strip() for n in str(x).split(",") if n.strip()]
    )
    print(f"[2/9]  Parsed node lists. Rows with nodes: {(df['node_list'].apply(len) > 0).sum()}")
    return df


# ---------------------------------------------------------------------------
# 3. Count node occurrences and keep top 30
# ---------------------------------------------------------------------------
def get_top_nodes(df: pd.DataFrame, top_n: int = TOP_N_NODES) -> set[str]:
    """Count all node occurrences and return the top_n most frequent."""
    all_nodes = [n for nodes in df["node_list"] for n in nodes]
    counter = Counter(all_nodes)
    top_nodes = {node for node, _ in counter.most_common(top_n)}
    print(f"[3/9]  Top {top_n} nodes selected from {len(counter)} unique node types")
    return top_nodes


# ---------------------------------------------------------------------------
# 4. Filter node lists to top 30 only
# ---------------------------------------------------------------------------
def filter_to_top_nodes(df: pd.DataFrame, top_nodes: set[str]) -> pd.DataFrame:
    """Keep only top-N nodes in each row's node list."""
    df = df.copy()
    df["node_list"] = df["node_list"].apply(
        lambda ns: [n for n in ns if n in top_nodes]
    )
    print(f"[4/9]  Filtered each row's nodes to top-{TOP_N_NODES} only")
    return df


# ---------------------------------------------------------------------------
# 5. Drop empty node lists
# ---------------------------------------------------------------------------
def drop_empty_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose filtered node list is empty."""
    before = len(df)
    df = df[df["node_list"].apply(len) > 0].reset_index(drop=True)
    print(f"[5/9]  Dropped {before - len(df)} rows with empty node lists  ->  {len(df)} remaining")
    return df


# ---------------------------------------------------------------------------
# 6. Binarize labels
# ---------------------------------------------------------------------------
def binarize_labels(df: pd.DataFrame) -> tuple[np.ndarray, MultiLabelBinarizer]:
    """Fit a MultiLabelBinarizer on the filtered node lists."""
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["node_list"])
    print(f"[6/9]  MultiLabelBinarizer fitted: {len(mlb.classes_)} node classes")
    return y, mlb


# ---------------------------------------------------------------------------
# 7. Train / test split
# ---------------------------------------------------------------------------
def split_data(
    X: pd.Series, y: np.ndarray
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """80/20 random split (no stratify for multi-label)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[7/9]  Split -> Train: {len(X_train)}  |  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 8 & 9. Build pipeline and fit
# ---------------------------------------------------------------------------
def build_and_train(
    X_train: pd.Series, y_train: np.ndarray
) -> Pipeline:
    """Build TF-IDF + OneVsRest(LR) pipeline and fit."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=500,
                C=1.0,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )
        )),
    ])
    print(
        "[8/9]  Pipeline built: "
        "TfidfVectorizer(max_features=5000, ngram_range=(1,2)) + "
        "OneVsRestClassifier(LogisticRegression(max_iter=500, C=1.0))"
    )
    pipeline.fit(X_train, y_train)
    print("[9/9]  Training complete")
    return pipeline


# ---------------------------------------------------------------------------
# 10. Evaluate
# ---------------------------------------------------------------------------
def evaluate(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: np.ndarray,
) -> None:
    """Print all required evaluation metrics."""
    y_pred = pipeline.predict(X_test)

    h_loss = hamming_loss(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_test, y_pred)   # exact-match (subset accuracy)

    print(f"\n[Evaluation on test set — {len(y_test)} samples]")
    print(f"  Hamming Loss          : {h_loss:.4f}")
    print(f"  F1 Score (micro)      : {f1_micro:.4f}")
    print(f"  F1 Score (macro)      : {f1_macro:.4f}")
    print(f"  F1 Score (weighted)   : {f1_weighted:.4f}")
    print(f"  Exact-Match Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")


# ---------------------------------------------------------------------------
# 11 & 12. Save artefacts
# ---------------------------------------------------------------------------
def save_artefacts(pipeline: Pipeline, mlb: MultiLabelBinarizer) -> None:
    """Save the trained pipeline and MLB with joblib."""
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    print(f"\n  node_model.pkl  saved ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"  mlb.pkl         saved ({MLB_PATH.stat().st_size:,} bytes)")
    print("\nModel saved successfully")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry-point: full training pipeline."""
    print("=" * 60)
    print("  Node Recommender -- Training (Multi-Label)")
    print("=" * 60 + "\n")

    df = load_data()
    df = parse_nodes(df)

    top_nodes = get_top_nodes(df)
    df = filter_to_top_nodes(df, top_nodes)
    df = drop_empty_nodes(df)

    y, mlb = binarize_labels(df)
    X = df["description"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline = build_and_train(X_train, y_train)
    evaluate(pipeline, X_test, y_test)
    save_artefacts(pipeline, mlb)


if __name__ == "__main__":
    main()
