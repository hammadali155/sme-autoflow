"""
main.py -- FastAPI Backend for SME AutoFlow.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

Exposes a REST API that receives a user's workflow description and returns:
  1. Predicted intent (from the intent classifier)
  2. Predicted nodes (from the node recommender)
  3. Similar templates (from the RAG retriever)
  4. A generated n8n workflow JSON (from Gemini)

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure project root is in sys.path for imports
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import retrieve  # noqa: E402
from generator.gemini_generator import generate_workflow  # noqa: E402

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
CLASSIFIER_DIR: Path = PROJECT_ROOT / "models" / "intent_classifier"
RECOMMENDER_DIR: Path = PROJECT_ROOT / "models" / "node_recommender"

# ---------------------------------------------------------------------------
# FastAPI app with lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load ML models at startup, release at shutdown."""
    # --- Startup ---
    print("[startup] Loading ML models...")
    application.state.intent_pipeline = joblib.load(CLASSIFIER_DIR / "intent_model.pkl")
    application.state.intent_encoder = joblib.load(CLASSIFIER_DIR / "label_encoder.pkl")
    application.state.node_pipeline = joblib.load(RECOMMENDER_DIR / "node_model.pkl")
    application.state.node_mlb = joblib.load(RECOMMENDER_DIR / "mlb.pkl")
    application.state.models_loaded = True
    print("[startup] All models loaded successfully.")
    yield
    # --- Shutdown ---
    print("[shutdown] Cleaning up...")
    application.state.models_loaded = False


app = FastAPI(
    title="SME AutoFlow API",
    description=(
        "AI-powered n8n workflow generator for SMEs. "
        "Accepts a natural-language business need and returns a complete "
        "n8n workflow JSON — powered by ML classifiers, RAG retrieval, "
        "and Google Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class WorkflowRequest(BaseModel):
    """User's natural-language workflow description."""
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        example="When a new lead fills out our contact form, send a welcome email via Gmail and notify sales on Slack.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of similar templates to retrieve for context.",
    )


class PredictionResponse(BaseModel):
    """ML prediction results (without Gemini generation)."""
    intent: str
    intent_confidence: float
    predicted_nodes: list[str]
    similar_templates: list[dict[str, Any]]


class WorkflowResponse(BaseModel):
    """Full response including generated workflow."""
    intent: str
    intent_confidence: float
    predicted_nodes: list[str]
    similar_templates: list[dict[str, Any]]
    workflow_json: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper: run ML predictions
# ---------------------------------------------------------------------------
def _predict(prompt: str, top_k: int = 3) -> dict[str, Any]:
    """Run intent classification, node recommendation, and RAG retrieval.

    Args:
        prompt: User's workflow description.
        top_k: Number of similar templates to retrieve.

    Returns:
        Dict with intent, confidence, nodes, and similar templates.
    """
    # Intent classification
    intent_idx = app.state.intent_pipeline.predict([prompt])[0]
    intent_label = app.state.intent_encoder.inverse_transform([intent_idx])[0]

    # Confidence score
    proba = app.state.intent_pipeline.predict_proba([prompt])[0]
    confidence = float(np.max(proba))

    # Node recommendation
    node_pred = app.state.node_pipeline.predict([prompt])[0]
    predicted_nodes = list(
        app.state.node_mlb.inverse_transform(node_pred.reshape(1, -1))[0]
    )

    # RAG retrieval
    similar = retrieve(prompt, top_k=top_k)

    return {
        "intent": intent_label,
        "intent_confidence": round(confidence, 4),
        "predicted_nodes": predicted_nodes,
        "similar_templates": similar,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check — confirms the API and ML models are operational."""
    return {
        "status": "ok",
        "models_loaded": getattr(app.state, "models_loaded", False),
        "service": "SME AutoFlow API",
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["ML"])
async def predict(request: WorkflowRequest) -> PredictionResponse:
    """Run ML predictions only (no Gemini call).

    Returns the detected intent, predicted nodes, and similar templates.
    Useful for previewing what the system detects *before* generating.
    """
    try:
        result = _predict(request.prompt, request.top_k)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.post("/generate", response_model=WorkflowResponse, tags=["Generator"])
async def generate(request: WorkflowRequest) -> WorkflowResponse:
    """Full pipeline: predict + generate workflow via Gemini.

    1. Classify intent
    2. Predict nodes
    3. Retrieve similar templates via RAG
    4. Generate workflow with Gemini 2.0 Flash
    5. Return complete WorkflowResponse
    """
    # Step 1-3: ML predictions + RAG
    try:
        result = _predict(request.prompt, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    # Step 4: Gemini generation
    template_jsons = [
        json.dumps(t, ensure_ascii=False) for t in result["similar_templates"]
    ]

    try:
        workflow = generate_workflow(
            user_request=request.prompt,
            predicted_intent=result["intent"],
            predicted_nodes=result["predicted_nodes"],
            similar_templates=template_jsons,
        )
    except EnvironmentError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini API not configured: {exc}",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned invalid response: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow generation failed: {exc}",
        )

    # Step 5: Return response
    return WorkflowResponse(
        **result,
        workflow_json=workflow,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Starting SME AutoFlow API on http://localhost:8000")
    print("Interactive docs at http://localhost:8000/docs")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
