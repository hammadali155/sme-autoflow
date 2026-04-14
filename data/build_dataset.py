"""
build_dataset.py -- Build labelled dataset from raw n8n template JSON files.

Reads every JSON file in data/raw_templates/, extracts description and node
list, auto-labels each template with an intent category based on which n8n
nodes are present, and writes two output files:

  1. data/processed_templates.json  -- full records (including raw JSON)
  2. data/labeled_dataset.csv       -- description, nodes, intent

Usage:
    python data/build_dataset.py
"""

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR: Path = Path(__file__).resolve().parent
RAW_DIR: Path = SCRIPT_DIR / "raw_templates"
PROCESSED_JSON: Path = SCRIPT_DIR / "processed_templates.json"
LABELED_CSV: Path = SCRIPT_DIR / "labeled_dataset.csv"

# ---------------------------------------------------------------------------
# Intent-labelling rules
# ---------------------------------------------------------------------------
# Each tuple: (set of keyword fragments to match against node type, label)
# Order matters -- first match wins.
INTENT_RULES: list[tuple[set[str], str]] = [
    ({"gmail", "email", "smtp", "imap"},                          "email_automation"),
    ({"slack", "telegram", "discord", "mattermost"},              "team_communication"),
    ({"webhook", "http", "rest", "graphql"},                      "data_sync"),
    ({"postgres", "mysql", "mongodb", "supabase", "airtable"},    "database_ops"),
    ({"twitter", "instagram", "linkedin", "facebook"},            "social_media"),
    ({"google_sheets", "notion", "trello", "asana", "clickup"},   "productivity"),
    ({"stripe", "shopify", "woocommerce", "paypal"},              "ecommerce"),
    ({"openai", "gemini", "huggingface", "cohere"},               "ai_tasks"),
    ({"schedule", "cron"},                                        "scheduling"),
]

DEFAULT_INTENT: str = "general"


def classify_intent(node_types: list[str]) -> str:
    """Return an intent label for a template based on its node types.

    The node type strings (e.g. 'n8n-nodes-base.gmail') are lowered and
    checked against each rule's keyword set.  The first rule that matches
    any node wins.

    Args:
        node_types: List of raw node-type strings from the template.

    Returns:
        The intent category string.
    """
    # Build a single lowered blob for fast substring matching.
    blob: str = " ".join(n.lower() for n in node_types)

    for keywords, label in INTENT_RULES:
        for kw in keywords:
            if kw in blob:
                return label

    return DEFAULT_INTENT


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_raw_templates() -> list[dict[str, Any]]:
    """Read all JSON files from the raw_templates directory.

    Returns:
        A list of parsed template dicts sorted by id.
    """
    templates: list[dict[str, Any]] = []
    json_files = sorted(RAW_DIR.glob("*.json"))

    if not json_files:
        print(f"[WARN] No JSON files found in {RAW_DIR}")
        return templates

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                templates.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping {fp.name}: {exc}")

    return templates


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_records(templates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enrich each template with an intent label and return full records.

    Args:
        templates: Raw template dicts.

    Returns:
        List of enriched dicts with keys:
            id, name, description, nodes, tags, intent, raw_json
    """
    records: list[dict[str, Any]] = []

    for tmpl in templates:
        nodes: list[str] = tmpl.get("nodes", [])
        description: str = tmpl.get("description", "")

        # Skip templates with no description -- they are not useful for ML.
        if not description.strip():
            continue

        intent: str = classify_intent(nodes)

        records.append({
            "id": tmpl.get("id"),
            "name": tmpl.get("name", ""),
            "description": description,
            "nodes": nodes,
            "tags": tmpl.get("tags", []),
            "intent": intent,
            "raw_json": tmpl,
        })

    return records


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def save_processed_json(records: list[dict[str, Any]]) -> None:
    """Write the full enriched records to processed_templates.json."""
    with open(PROCESSED_JSON, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"[SAVED] {PROCESSED_JSON.name}  ({len(records)} records)")


def save_labeled_csv(records: list[dict[str, Any]]) -> None:
    """Write the slim labelled dataset to labeled_dataset.csv."""
    with open(LABELED_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["description", "nodes", "intent"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "description": rec["description"],
                "nodes": ",".join(rec["nodes"]),
                "intent": rec["intent"],
            })
    print(f"[SAVED] {LABELED_CSV.name}  ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: load, label, save, and print summary stats."""
    print("[START] Loading raw templates ...\n")
    templates = load_raw_templates()
    print(f"  Loaded {len(templates)} raw template files.\n")

    records = build_records(templates)
    print(f"  Built {len(records)} labelled records "
          f"(skipped {len(templates) - len(records)} with empty description).\n")

    save_processed_json(records)
    save_labeled_csv(records)

    # --- Summary ---
    print("\n--- Intent Category Breakdown ---")
    df = pd.DataFrame(records)
    counts = df["intent"].value_counts()
    print(counts.to_string())
    print(f"\nTotal: {len(df)}")


if __name__ == "__main__":
    main()
