"""
test_api.py -- Integration tests for the FastAPI backend.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project

Uses FastAPI TestClient (runs without a live server).
Both the RAG retriever and Gemini generator are mocked
so no network calls are made and no GPU/HF model loading happens.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------
MOCK_WORKFLOW = {
    "name": "Test Workflow",
    "nodes": [
        {
            "name": "Form Trigger",
            "type": "n8n-nodes-base.formTrigger",
            "typeVersion": 1,
            "position": [250, 300],
            "parameters": {},
        },
        {
            "name": "Send Email",
            "type": "n8n-nodes-base.gmail",
            "typeVersion": 1,
            "position": [500, 300],
            "parameters": {"to": "={{$json.email}}"},
        },
    ],
    "connections": {
        "Form Trigger": {
            "main": [[{"node": "Send Email", "type": "main", "index": 0}]]
        }
    },
}

MOCK_SIMILAR_TEMPLATES = [
    {
        "id": "tpl_0",
        "distance": 0.42,
        "name": "Gmail form handler",
        "description": "Handles form submissions via Gmail",
        "nodes": ["n8n-nodes-base.formTrigger", "n8n-nodes-base.gmail"],
        "tags": [],
    }
]


# ---------------------------------------------------------------------------
# Shared fixture: TestClient with both RAG and Gemini mocked
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """TestClient with RAG retriever and Gemini generator both mocked."""
    with patch(
        "rag.retriever.retrieve",
        return_value=MOCK_SIMILAR_TEMPLATES,
    ), patch(
        "generator.gemini_generator.generate_workflow",
        return_value=MOCK_WORKFLOW,
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    """GET /health must return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_status_ok(client):
    """GET /health must include {"status": "ok"}."""
    data = client.get("/health").json()
    assert data.get("status") == "ok"


def test_health_models_loaded(client):
    """GET /health must report models_loaded: true after startup."""
    data = client.get("/health").json()
    assert data.get("models_loaded") is True


# ---------------------------------------------------------------------------
# Tests: POST /generate — valid prompts
# ---------------------------------------------------------------------------

def test_generate_returns_200(client):
    """POST /generate with a valid prompt must return HTTP 200."""
    resp = client.post(
        "/generate",
        json={"prompt": "Send a Slack message when a payment comes in"},
    )
    assert resp.status_code == 200, f"Unexpected: {resp.json()}"


def test_generate_response_has_intent(client):
    """Response must contain 'intent' as a string."""
    data = client.post(
        "/generate",
        json={"prompt": "Send a Slack message when a payment comes in"},
    ).json()
    assert "intent" in data, f"Missing 'intent': {data}"
    assert isinstance(data["intent"], str)


def test_generate_response_has_predicted_nodes(client):
    """Response must contain 'predicted_nodes' as a list."""
    data = client.post(
        "/generate",
        json={"prompt": "Send a welcome email via Gmail after form submission"},
    ).json()
    assert "predicted_nodes" in data, f"Missing 'predicted_nodes': {data}"
    assert isinstance(data["predicted_nodes"], list)


def test_generate_response_has_workflow_json(client):
    """Response must contain 'workflow_json' as a dict."""
    data = client.post(
        "/generate",
        json={"prompt": "Use OpenAI to summarise incoming support tickets"},
    ).json()
    assert "workflow_json" in data, f"Missing 'workflow_json': {data}"
    assert isinstance(data["workflow_json"], dict)


def test_generate_workflow_json_has_nodes(client):
    """The workflow_json must contain a 'nodes' list."""
    data = client.post(
        "/generate",
        json={"prompt": "Sync Postgres records to Google Sheets every hour"},
    ).json()
    workflow = data.get("workflow_json", {})
    assert "nodes" in workflow, f"workflow_json missing 'nodes': {workflow}"
    assert isinstance(workflow["nodes"], list)


def test_generate_response_has_intent_confidence(client):
    """Response must include 'intent_confidence' as a float in [0, 1]."""
    data = client.post(
        "/generate",
        json={"prompt": "Post daily news to Telegram group"},
    ).json()
    assert "intent_confidence" in data, f"Missing 'intent_confidence': {data}"
    assert 0.0 <= data["intent_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: POST /generate — invalid / edge-case prompts
# ---------------------------------------------------------------------------

def test_generate_empty_prompt_returns_error(client):
    """POST /generate with an empty string must return 422 (Pydantic min_length)."""
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422


def test_generate_short_prompt_returns_error(client):
    """POST /generate with a too-short prompt must return 422."""
    resp = client.post("/generate", json={"prompt": "hi"})
    assert resp.status_code == 422


def test_generate_missing_prompt_field_returns_error(client):
    """POST /generate with no 'prompt' field must return 422."""
    resp = client.post("/generate", json={})
    assert resp.status_code == 422
