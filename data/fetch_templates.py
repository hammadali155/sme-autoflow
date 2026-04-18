"""
fetch_templates.py — Fetch n8n workflow templates from the public API.

Iterates pages 1–20 (50 rows each) of the n8n template search endpoint,
extracts key fields from every template, and saves each one as a separate
JSON file under data/raw_templates/.

Usage:
    python data/fetch_templates.py
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL: str = "https://api.n8n.io/api/templates/search"
START_PAGE: int = 1
END_PAGE: int = 100
ROWS_PER_PAGE: int = 50
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 2.0      # base delay; doubles on each retry
REQUEST_TIMEOUT_SECONDS: int = 30
PROGRESS_INTERVAL: int = 5            # print progress every N pages

# Output directory (resolved relative to this script's location)
SCRIPT_DIR: Path = Path(__file__).resolve().parent
RAW_TEMPLATES_DIR: Path = SCRIPT_DIR / "raw_templates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_template(template: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields we care about from a raw API template object.

    Args:
        template: A single template dict returned by the n8n API.

    Returns:
        A trimmed dict containing id, name, description, nodes, and tags.
    """
    # Nodes may be nested in different ways depending on the API version.
    # We normalise to a flat list of node-type strings.
    raw_nodes: list[Any] = template.get("nodes", [])
    node_types: list[str] = []
    for node in raw_nodes:
        if isinstance(node, dict):
            node_type = node.get("type") or node.get("name", "")
            if node_type:
                node_types.append(node_type)
        elif isinstance(node, str):
            node_types.append(node)

    # Tags — similarly normalise to a list of strings.
    raw_tags: list[Any] = template.get("categories", []) or template.get("tags", [])
    tags: list[str] = []
    for tag in raw_tags:
        if isinstance(tag, dict):
            tags.append(tag.get("name", str(tag.get("id", ""))))
        elif isinstance(tag, str):
            tags.append(tag)

    return {
        "id": template.get("id"),
        "name": template.get("name", ""),
        "description": template.get("description", ""),
        "nodes": node_types,
        "tags": tags,
    }


def fetch_page(page: int) -> list[dict[str, Any]]:
    """Fetch a single page of templates with retry logic.

    Args:
        page: The 1-indexed page number to fetch.

    Returns:
        A list of extracted template dicts for that page.

    Raises:
        RuntimeError: If the request still fails after MAX_RETRIES attempts.
    """
    params: dict[str, int] = {"page": page, "rows": ROWS_PER_PAGE}
    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            # The API wraps results under a "workflows" key.
            workflows: list[dict[str, Any]] = data.get("workflows", [])
            return [_extract_template(w) for w in workflows]

        except requests.RequestException as exc:
            last_exception = exc
            delay = RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"  [WARN] Page {page} attempt {attempt}/{MAX_RETRIES} failed: "
                f"{exc}. Retrying in {delay:.1f}s ..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Failed to fetch page {page} after {MAX_RETRIES} retries. "
        f"Last error: {last_exception}"
    )


def save_template(template: dict[str, Any]) -> Path:
    """Persist a single template dict as a JSON file.

    Args:
        template: The extracted template dict (must contain an 'id' key).

    Returns:
        The Path to the saved file.
    """
    file_path: Path = RAW_TEMPLATES_DIR / f"template_{template['id']}.json"
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(template, fh, indent=2, ensure_ascii=False)
    return file_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: fetch pages, save templates, report progress."""
    RAW_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

    total_saved: int = 0
    seen_ids: set[int] = set()

    print(f"[START] Fetching n8n templates -- pages {START_PAGE}-{END_PAGE}, "
          f"{ROWS_PER_PAGE} rows/page ...\n")

    for page in range(START_PAGE, END_PAGE + 1):
        try:
            templates = fetch_page(page)
        except RuntimeError as exc:
            print(f"  [ERROR] Skipping page {page}: {exc}")
            continue

        page_saved: int = 0
        for tmpl in templates:
            tmpl_id = tmpl.get("id")
            if tmpl_id is None or tmpl_id in seen_ids:
                continue
            seen_ids.add(tmpl_id)
            save_template(tmpl)
            page_saved += 1

        total_saved += page_saved

        # Progress report every PROGRESS_INTERVAL pages
        if page % PROGRESS_INTERVAL == 0:
            print(f"  [PROGRESS] Page {page}/{END_PAGE} done -- "
                  f"{page_saved} new templates this page, "
                  f"{total_saved} total so far")

        # Be polite to the API
        time.sleep(0.3)

    print(f"\n[DONE] Finished! Saved {total_saved} unique templates to "
          f"{RAW_TEMPLATES_DIR.resolve()}")


if __name__ == "__main__":
    main()
