"""
embed_templates.py -- Index n8n templates into ChromaDB for RAG retrieval.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Loads processed_templates.json, encodes each template using
sentence-transformers (all-MiniLM-L6-v2), and stores them in a
persistent ChromaDB collection for later similarity search.

Usage:
    python rag/embed_templates.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_JSON: Path = DATA_DIR / "processed_templates.json"
CHROMA_DIR: Path = SCRIPT_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "n8n_templates"
MODEL_NAME: str = "all-MiniLM-L6-v2"
BATCH_SIZE: int = 100


# ---------------------------------------------------------------------------
# 1. Load templates
# ---------------------------------------------------------------------------
def load_templates() -> list[dict[str, Any]]:
    """Load processed_templates.json."""
    if not PROCESSED_JSON.exists():
        print(f"[ERROR] File not found: {PROCESSED_JSON}")
        print("       Run  python data/build_dataset.py  first.")
        sys.exit(1)

    with open(PROCESSED_JSON, "r", encoding="utf-8") as fh:
        templates = json.load(fh)

    print(f"[1/5] Loaded {len(templates)} templates from {PROCESSED_JSON.name}")
    return templates


# ---------------------------------------------------------------------------
# 2. Initialize sentence transformer
# ---------------------------------------------------------------------------
def load_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    print(f"[2/5] Loading SentenceTransformer('{MODEL_NAME}')...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"      Model loaded (dim={model.get_embedding_dimension()})")
    return model


# ---------------------------------------------------------------------------
# 3 & 4. Initialize ChromaDB
# ---------------------------------------------------------------------------
def init_chroma() -> tuple[chromadb.ClientAPI, chromadb.Collection]:
    """Create a persistent ChromaDB client and collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"[3/5] ChromaDB initialized at {CHROMA_DIR}")
    print(f"      Collection '{COLLECTION_NAME}' -- {collection.count()} existing docs")
    return client, collection


# ---------------------------------------------------------------------------
# 5-8. Embed and index in batches
# ---------------------------------------------------------------------------
def embed_and_index(
    templates: list[dict[str, Any]],
    model: SentenceTransformer,
    collection: chromadb.Collection,
) -> int:
    """Encode templates and upsert into ChromaDB in batches.

    Args:
        templates: List of processed template dicts.
        model: SentenceTransformer model.
        collection: ChromaDB collection.

    Returns:
        Number of templates indexed.
    """
    total = len(templates)
    indexed = 0

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = templates[start:end]

        # Build text for encoding: description + node info
        texts: list[str] = []
        documents: list[str] = []
        ids: list[str] = []

        for i, tmpl in enumerate(batch):
            desc = tmpl.get("description", "")
            nodes = ", ".join(tmpl.get("nodes", []))
            text = f"{desc} uses nodes: {nodes}"
            texts.append(text)
            documents.append(json.dumps(tmpl, ensure_ascii=False))
            ids.append(f"tpl_{start + i}")

        # Encode batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
        )

        indexed += len(batch)

        # Progress every batch
        print(f"[4/5] Batch {start // BATCH_SIZE + 1}: "
              f"indexed {indexed}/{total} templates")

    return indexed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry-point: load, embed, index."""
    print("=" * 60)
    print("  RAG -- Template Embedding & Indexing")
    print("=" * 60 + "\n")

    templates = load_templates()
    _, collection = init_chroma()

    # Guard: skip if already indexed
    if collection.count() >= len(templates):
        print(f"\n[SKIP] Collection already has {collection.count()} documents. "
              "Already indexed.")
        return

    model = load_model()
    indexed = embed_and_index(templates, model, collection)

    print(f"\n[5/5] Done! {indexed} templates indexed into ChromaDB")
    print(f"      Collection now has {collection.count()} documents")


if __name__ == "__main__":
    main()
