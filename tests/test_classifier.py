"""
test_classifier.py -- Unit tests for the intent classifier.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
"""

import sys
from pathlib import Path

import joblib
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSIFIER_DIR = PROJECT_ROOT / "models" / "intent_classifier"
KNOWN_CATEGORIES = [
    "ai_tasks",
    "data_sync",
    "database_ops",
    "email_automation",
    "general",
    "team_communication",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline():
    """Load the intent classifier pipeline once for the whole module."""
    return joblib.load(CLASSIFIER_DIR / "intent_model.pkl")


@pytest.fixture(scope="module")
def encoder():
    """Load the label encoder once for the whole module."""
    return joblib.load(CLASSIFIER_DIR / "label_encoder.pkl")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_file_exists():
    """Model pkl file must exist at the expected path."""
    assert (CLASSIFIER_DIR / "intent_model.pkl").exists(), (
        f"intent_model.pkl not found in {CLASSIFIER_DIR}"
    )


def test_encoder_file_exists():
    """Label encoder pkl file must exist at the expected path."""
    assert (CLASSIFIER_DIR / "label_encoder.pkl").exists(), (
        f"label_encoder.pkl not found in {CLASSIFIER_DIR}"
    )


def test_model_loads_without_error(pipeline):
    """Loading the model should produce a non-None pipeline."""
    assert pipeline is not None


def test_encoder_loads_without_error(encoder):
    """Loading the encoder should produce a non-None object."""
    assert encoder is not None


def test_encoder_classes(encoder):
    """Encoder classes must be a subset of known categories."""
    for cls in encoder.classes_:
        assert cls in KNOWN_CATEGORIES, f"Unknown class: {cls}"


@pytest.mark.parametrize("description,expected_type", [
    (
        "Send a Slack message when a new row is added to Google Sheets",
        str,
    ),
    (
        "Use OpenAI to classify incoming support tickets automatically",
        str,
    ),
    (
        "Send a welcome email via Gmail after a form submission",
        str,
    ),
    (
        "Sync customer records from Postgres to our CRM every hour",
        str,
    ),
    (
        "Post a daily news digest to our Telegram group",
        str,
    ),
])
def test_prediction_returns_known_category(pipeline, encoder, description, expected_type):
    """Each prediction must be a string from the known category list."""
    idx = pipeline.predict([description])[0]
    label = encoder.inverse_transform([idx])[0]

    assert isinstance(label, expected_type), (
        f"Expected str, got {type(label)}"
    )
    assert label in KNOWN_CATEGORIES, (
        f"Predicted '{label}' is not in known categories: {KNOWN_CATEGORIES}"
    )


def test_prediction_confidence_in_range(pipeline):
    """predict_proba output must sum to ~1.0 and all values in [0, 1]."""
    proba = pipeline.predict_proba(["Automate my email workflow"])[0]
    assert abs(sum(proba) - 1.0) < 1e-5, "Probabilities must sum to 1"
    assert all(0.0 <= p <= 1.0 for p in proba), "All probs must be in [0, 1]"
