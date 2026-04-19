"""
streamlit_app.py -- Streamlit Frontend for SME AutoFlow.
SME AutoFlow | GDGoC AI/ML Fellowship Final Project
Author: Hammad Ali (FA23-BCS-007)

A polished Streamlit UI that connects to the FastAPI backend to:
  1. Accept a natural-language business requirement
  2. Display ML predictions (intent + nodes)
  3. Render the generated n8n workflow JSON
  4. Optionally deploy the workflow to a live n8n instance

Usage:
    streamlit run app/streamlit_app.py
"""

import json
import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SME AutoFlow ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        margin-top: -8px;
        margin-bottom: 30px;
    }

    /* Intent badge */
    .intent-badge {
        display: inline-block;
        padding: 8px 20px;
        background: linear-gradient(135deg, #00c853, #1de9b6);
        color: white;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }

    /* Node chips */
    .node-chip {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        background: #e8eaf6;
        color: #3949ab;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Success checkmark */
    .check-circle {
        font-size: 3rem;
        color: #00c853;
    }

    /* Quick-fill buttons */
    .stButton > button {
        border-radius: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    API_BASE_URL = st.secrets.get("API_BASE_URL") or os.environ.get("API_BASE_URL", "http://localhost:8000")
except Exception:
    API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

EXAMPLE_PROMPTS = [
    "Send a Slack notification when a new payment is received in Stripe",
    "When a Google Form is submitted, save the data to Google Sheets and send a confirmation email via Gmail",
    "Use OpenAI to summarise new Zendesk tickets and post a daily digest to our Telegram group",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### n8n Deployment (Optional)")
    try:
        default_n8n_url = st.secrets.get("N8N_URL", "")
        default_n8n_key = st.secrets.get("N8N_API_KEY", "")
    except Exception:
        default_n8n_url = ""
        default_n8n_key = ""

    n8n_url = st.text_input(
        "n8n Instance URL",
        value=default_n8n_url,
        placeholder="https://your-n8n.example.com",
        help="Base URL of your n8n instance (e.g. https://n8n.example.com)",
    )
    n8n_api_key = st.text_input(
        "n8n API Key",
        value=default_n8n_key,
        type="password",
        placeholder="n8n-api-key-here",
        help="API key from your n8n instance Settings → API",
    )

    st.caption(
        "💡 **Leave blank** to preview the generated workflow without deploying. "
        "You can always copy the JSON and import it manually."
    )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**SME AutoFlow** uses ML classifiers, semantic search (RAG), "
        "and Google Gemini to generate production-ready n8n workflows "
        "from plain English."
    )
    st.markdown(
        "Built by **Hammad Ali** (FA23-BCS-007)  \n"
        "GDGoC AI/ML Fellowship Final Project"
    )

# ---------------------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">SME AutoFlow ⚡</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered n8n Workflow Generator — '
    'Describe your business problem, get a ready-to-deploy workflow</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
prompt = st.text_area(
    "💬 Describe your business problem",
    placeholder="e.g. Send a Slack message when I receive a payment in Stripe",
    height=120,
    key="main_prompt",
)

# Quick-fill example buttons
st.markdown("**Try an example:**")

def set_prompt(text):
    st.session_state["main_prompt"] = text

cols_ex = st.columns(3)
for i, example in enumerate(EXAMPLE_PROMPTS):
    with cols_ex[i]:
        st.button(
            f"📝 Example {i + 1}", 
            key=f"example_{i}", 
            use_container_width=True, 
            on_click=set_prompt, 
            args=(example,)
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------
generate_clicked = st.button(
    "⚡ Generate Workflow",
    type="primary",
    use_container_width=True,
    disabled=not prompt or len(prompt.strip()) < 10,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def call_api(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the FastAPI backend and return the JSON response."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(
            "🔌 **Cannot reach the API server.** "
            "Make sure the FastAPI backend is running:\n\n"
            "```\nuvicorn api.main:app --reload --port 8000\n```"
        )
        st.stop()
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        st.error(f"❌ **API Error ({exc.response.status_code}):** {detail}")
        st.stop()
    except requests.Timeout:
        st.error(
            "⏱️ **Request timed out.** The Gemini API may be slow. "
            "Please try again."
        )
        st.stop()


def deploy_to_n8n(
    workflow_json: dict[str, Any], base_url: str, api_key: str
) -> dict[str, Any]:
    """Deploy the generated workflow to a live n8n instance."""
    base_url = base_url.strip()
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "http://" + base_url
        
    url = f"{base_url.rstrip('/')}/api/v1/workflows"
    headers = {"X-N8N-API-KEY": api_key, "Content-Type": "application/json"}

    # Strictly filter out hallucinated node properties that violate the n8n API schema
    valid_node_keys = {"id", "name", "type", "typeVersion", "position", "parameters", "webhookId", "notes", "credentials", "executeOnce", "alwaysOutputData", "retryOnFail", "continueOnFail"}
    cleaned_nodes = []
    for node in workflow_json.get("nodes", []):
        cleaned_node = {k: v for k, v in node.items() if k in valid_node_keys}
        cleaned_nodes.append(cleaned_node)

    payload = {
        "name": workflow_json.get("name", "SME AutoFlow Generated Workflow"),
        "nodes": cleaned_nodes,
        "connections": workflow_json.get("connections", {}),
        "settings": workflow_json.get("settings", {}),
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------
if generate_clicked and prompt and len(prompt.strip()) >= 10:
    with st.spinner("🤖 Classifying intent and generating workflow..."):
        st.session_state["generation_result"] = call_api("/generate", {"prompt": prompt.strip()})

if "generation_result" in st.session_state:
    result = st.session_state["generation_result"]

    # --- Success: show results ---
    st.success("✅ Workflow generated successfully!")

    # --- Three-column summary ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Detected Intent")
        intent = result.get("intent", "unknown")
        confidence = result.get("intent_confidence", 0)
        st.markdown(
            f'<span class="intent-badge">{intent}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Confidence: {confidence:.1%}")

    with col2:
        st.markdown("#### 🧩 Recommended Nodes")
        nodes = result.get("predicted_nodes", [])
        if nodes:
            chips_html = "".join(
                f'<span class="node-chip">{n.split(".")[-1]}</span>' for n in nodes
            )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("No specific nodes predicted")

    with col3:
        st.markdown("#### ✅ Workflow Ready")
        st.markdown('<div class="check-circle">✓</div>', unsafe_allow_html=True)
        workflow = result.get("workflow_json", {})
        node_count = len(workflow.get("nodes", []))
        st.caption(f"{node_count} nodes in workflow")

    st.markdown("---")

    # --- Similar templates (expandable) ---
    similar = result.get("similar_templates", [])
    if similar:
        with st.expander(f"📚 Similar Templates Used for Context ({len(similar)})", expanded=False):
            for i, tmpl in enumerate(similar, 1):
                name = tmpl.get("name", "Unnamed")
                dist = tmpl.get("distance", 0)
                tmpl_nodes = tmpl.get("nodes", [])
                st.markdown(
                    f"**{i}. {name}** (distance: {dist:.4f})  \n"
                    f"Nodes: {', '.join(n.split('.')[-1] for n in tmpl_nodes[:6])}"
                )

    # --- Workflow JSON ---
    st.markdown("### 📋 Generated Workflow JSON")
    workflow_str = json.dumps(workflow, indent=2, ensure_ascii=False)
    st.code(workflow_str, language="json")

    # --- Download button ---
    st.download_button(
        label="💾 Download Workflow JSON",
        data=workflow_str,
        file_name="sme_autoflow_workflow.json",
        mime="application/json",
        use_container_width=True,
    )

    # --- Deploy to n8n ---
    if n8n_url and n8n_api_key:
        st.markdown("---")
        st.markdown("### 🚀 Deploy to n8n")
        if st.button("Deploy to n8n ⚡", type="primary", use_container_width=True):
            with st.spinner("Deploying to n8n..."):
                try:
                    deploy_result = deploy_to_n8n(workflow, n8n_url, n8n_api_key)
                    wf_id = deploy_result.get("id", "unknown")
                    st.success(
                        f"🎉 **Workflow deployed successfully!**  \n"
                        f"Workflow ID: `{wf_id}`  \n"
                        f"Open it at: {n8n_url.rstrip('/')}/workflow/{wf_id}"
                    )
                    st.balloons()
                except requests.HTTPError as exc:
                    detail = ""
                    try:
                        detail = exc.response.json().get("message", str(exc))
                    except Exception:
                        detail = str(exc)
                    st.error(f"❌ **Deployment failed:** {detail}")
                except requests.ConnectionError:
                    st.error(
                        f"🔌 Cannot connect to n8n at `{n8n_url}`. "
                        "Check the URL and make sure the instance is running."
                    )
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")
    elif n8n_url or n8n_api_key:
        st.warning(
            "⚠️ Both **n8n URL** and **API Key** are required for deployment. "
            "Fill in both fields in the sidebar."
        )
