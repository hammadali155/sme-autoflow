"""
retriever.py -- RAG Retriever for n8n workflow templates.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Queries the ChromaDB vector store to find the most similar workflow
templates to a user's natural-language description.

Usage (standalone test):
    python rag/retriever.py
"""

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
CHROMA_DIR: Path = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME: str = "n8n_templates"
MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_TOP_K: int = 3

# ---------------------------------------------------------------------------
# Module-level singletons (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    """Lazy-load the ChromaDB collection."""
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB directory not found at {CHROMA_DIR}. "
                "Run  python rag/embed_templates.py  first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Find the top-k most similar templates to a query string.

    Args:
        query:  Natural-language workflow description from the user.
        top_k:  Number of results to return (default 3).

    Returns:
        List of dicts, each containing:
            - id         (str)   ChromaDB document ID
            - distance   (float) Lower = more similar
            - name       (str)   Template name
            - description(str)   Template description
            - nodes      (list)  List of node type strings
            - tags       (list)  Template tags
    """
    model = _get_model()
    collection = _get_collection()

    # Encode the query
    query_embedding = model.encode([query]).tolist()[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )

    # Parse and return results
    retrieved: list[dict[str, Any]] = []
    for doc_json, dist, doc_id in zip(
        results["documents"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        try:
            doc = json.loads(doc_json)
        except (json.JSONDecodeError, TypeError):
            doc = {}

        retrieved.append({
            "id": doc_id,
            "distance": round(dist, 4),
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "nodes": doc.get("nodes", []),
            "tags": doc.get("tags", []),
        })

    return retrieved


def format_context(templates: list[dict[str, Any]]) -> str:
    """Format retrieved templates into a compact string for LLM context injection.

    Args:
        templates: List of retrieved template dicts (from ``retrieve``).

    Returns:
        Multi-line string summarising each template's name, description,
        and nodes, ready to be injected into a Gemini prompt.
    """
    if not templates:
        return "No similar templates found."

    lines: list[str] = ["=== Similar n8n Workflow Templates ===\n"]
    for i, tmpl in enumerate(templates, start=1):
        nodes_str = ", ".join(tmpl["nodes"][:10]) if tmpl["nodes"] else "N/A"
        lines.append(
            f"[Template {i}] {tmpl['name']}\n"
            f"  Similarity distance : {tmpl['distance']}\n"
            f"  Description         : {tmpl['description'][:200]}...\n"
            f"  Key nodes           : {nodes_str}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_queries = [
        "Send a Slack message when a new row is added to Google Sheets",
        "Use OpenAI to classify incoming emails and route them automatically",
        "Sync new Postgres records to an external REST API every hour",
    ]

    print("=" * 60)
    print("  RAG Retriever -- Smoke Test")
    print("=" * 60)

    for q in test_queries:
        print(f"\n[Query] {q}")
        print("-" * 50)
        try:
            results = retrieve(q, top_k=3)
            for r in results:
                print(f"  [{r['distance']:.4f}] {r['name']}")
                print(f"           nodes: {', '.join(r['nodes'][:5])}")
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    print("\n[OK] Retriever working correctly.")
