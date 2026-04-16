"""
gemini_generator.py -- Gemini-powered n8n Workflow Generator.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Takes ML predictions (intent, nodes) and RAG-retrieved templates, then
prompts Google Gemini 2.0 Flash to generate a complete, valid n8n
workflow JSON.

Usage (standalone test):
    python generator/gemini_generator.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
MODEL_NAME: str = "gemini-2.0-flash"
MAX_RETRIES: int = 2
MAX_TEMPLATE_CHARS: int = 800
MAX_REFERENCE_TEMPLATES: int = 3

# ---------------------------------------------------------------------------
# Gemini client singleton
# ---------------------------------------------------------------------------
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazy-initialize and return the Gemini client (singleton)."""
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not found.  "
            "Create a .env file in the project root with:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "Or export it as an environment variable."
        )
    _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def _build_prompt(
    user_request: str,
    predicted_intent: str,
    predicted_nodes: list[str],
    similar_templates: list[str],
) -> str:
    """Construct a detailed system + user prompt for Gemini.

    Args:
        user_request:      Free-text workflow description from the user.
        predicted_intent:  Intent category (e.g. ``email_automation``).
        predicted_nodes:   List of recommended n8n node type strings.
        similar_templates: Raw JSON strings of RAG-retrieved templates.

    Returns:
        A single prompt string ready for ``model.generate_content()``.
    """
    # Truncate and format reference templates
    ref_blocks: list[str] = []
    for i, raw_json in enumerate(similar_templates[:MAX_REFERENCE_TEMPLATES], 1):
        try:
            tmpl = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            name = tmpl.get("name", "Unnamed")
            desc = tmpl.get("description", "")[:200]
            nodes = tmpl.get("nodes", [])
            snippet = json.dumps(tmpl, ensure_ascii=False)[:MAX_TEMPLATE_CHARS]
            ref_blocks.append(
                f"--- Reference Template {i}: {name} ---\n"
                f"Description: {desc}...\n"
                f"Nodes: {', '.join(nodes[:8])}\n"
                f"JSON snippet:\n{snippet}\n"
            )
        except (json.JSONDecodeError, TypeError):
            continue

    ref_section = "\n".join(ref_blocks) if ref_blocks else "No reference templates available."
    nodes_str = ", ".join(predicted_nodes) if predicted_nodes else "Let the model decide"

    prompt = f"""You are an expert n8n workflow automation builder. Your task is to generate
a complete, valid n8n workflow JSON based on the user's business requirement.

=== CONTEXT FROM ML MODELS ===

Detected Intent Category: {predicted_intent}
Recommended n8n Nodes: {nodes_str}

=== SIMILAR EXISTING WORKFLOWS (for reference) ===

{ref_section}

=== USER'S REQUEST ===

"{user_request}"

=== INSTRUCTIONS ===

1. Generate a COMPLETE, VALID n8n workflow JSON that fulfills the user's request.
2. Use the recommended nodes above as guidance, but add or remove nodes as needed
   for a working workflow.
3. Include proper node connections in the "connections" object.
4. Each node must have:
   - A unique "name"
   - The correct "type" (e.g. "n8n-nodes-base.gmail")
   - A "typeVersion" (use the latest stable version)
   - A "position" array [x, y] with reasonable layout coordinates
   - A "parameters" object (can contain placeholder values like "={{{{$json.email}}}}")
5. Include a trigger node as the first node (e.g. webhook, schedule, or form trigger).
6. The workflow should be realistic and production-ready in structure.

=== OUTPUT FORMAT ===

Return ONLY the raw n8n workflow JSON object. Do NOT include:
- Any explanatory text before or after the JSON
- Markdown code fences (```)
- Comments inside the JSON

Start your response with {{ and end with }}."""

    return prompt


# ---------------------------------------------------------------------------
# Response cleaner
# ---------------------------------------------------------------------------
def _clean_json_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences and parse the Gemini response as JSON.

    Args:
        raw: Raw text response from Gemini.

    Returns:
        Parsed workflow dict.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    text = raw.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try to extract JSON object if there's text around it
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        workflow = json.loads(text)
    except json.JSONDecodeError as exc:
        # Show first 500 chars of the bad response for debugging
        preview = text[:500]
        raise ValueError(
            f"Gemini returned invalid JSON. Parse error: {exc}\n"
            f"Response preview:\n{preview}"
        ) from exc

    return workflow


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_workflow(
    user_request: str,
    predicted_intent: str,
    predicted_nodes: list[str],
    similar_templates: list[str],
    *,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Generate a complete n8n workflow JSON using Google Gemini.

    Args:
        user_request:      Natural-language description of the desired workflow.
        predicted_intent:  Intent label from the classifier.
        predicted_nodes:   Node types from the recommender.
        similar_templates: Raw JSON strings of retrieved templates.
        temperature:       Gemini sampling temperature (0.0-2.0, default 0.7).

    Returns:
        Parsed n8n workflow as a Python dict.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
        ValueError:       If the response is not valid JSON after retries.
        Exception:        On Gemini API errors after retries.
    """
    client = _get_client()

    prompt = _build_prompt(
        user_request, predicted_intent, predicted_nodes, similar_templates
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=4096,
                ),
            )
            raw_text = response.text
            workflow = _clean_json_response(raw_text)

            # Basic sanity check
            if "nodes" not in workflow and "name" not in workflow:
                raise ValueError("Response JSON lacks 'nodes' or 'name' — probably not a valid workflow.")

            return workflow

        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                print(f"  [WARN] Attempt {attempt} returned bad JSON, retrying...")
                time.sleep(1)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                print(f"  [WARN] API error on attempt {attempt}: {exc}. Retrying...")
                time.sleep(2)

    raise ValueError(
        f"Failed to generate valid workflow JSON after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Gemini Workflow Generator -- Smoke Test")
    print("=" * 60 + "\n")

    test_request = (
        "When a new lead fills out our contact form, send them a welcome "
        "email via Gmail and notify our sales team on Slack."
    )
    test_intent = "email_automation"
    test_nodes = [
        "n8n-nodes-base.formTrigger",
        "n8n-nodes-base.gmail",
        "n8n-nodes-base.slack",
    ]
    test_templates: list[str] = []  # no RAG templates for smoke test

    print(f"Request : {test_request}")
    print(f"Intent  : {test_intent}")
    print(f"Nodes   : {test_nodes}\n")

    try:
        workflow = generate_workflow(
            user_request=test_request,
            predicted_intent=test_intent,
            predicted_nodes=test_nodes,
            similar_templates=test_templates,
        )
        print("[OK] Workflow generated successfully!")
        print(f"     Keys: {list(workflow.keys())}")
        if "nodes" in workflow:
            print(f"     Nodes: {len(workflow['nodes'])}")
            for node in workflow["nodes"]:
                print(f"       - {node.get('name', '?')} ({node.get('type', '?')})")
        print(f"\n     Full JSON preview (first 800 chars):")
        print(json.dumps(workflow, indent=2, ensure_ascii=False)[:800])
    except EnvironmentError as exc:
        print(f"[SKIP] {exc}")
        sys.exit(0)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
