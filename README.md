# SME AutoFlow ⚡

**AI-powered n8n workflow generator that turns plain English business requirements into deployable automation pipelines.**

## What It Does
SME AutoFlow bridges the gap between non-technical business owners and sophisticated automation tools. A user simply types what they want to achieve (e.g., "Send a Slack message when a new lead fills out a Google Form"). The system then runs natural language through a multi-stage ML pipeline to classify the intent, recommend relevant n8n nodes, and retrieve contextually similar, expert-built templates via RAG. Finally, Google Gemini processes this rich context to generate a highly accurate, fully-formed JSON workflow that can be immediately deployed to a live n8n instance via the integrated Streamlit UI.

## Demo
[Streamlit App Placeholder - Insert deployment link here]

## Architecture

```text
User Input ("When a new lead fills form...")
         │
         ├──► 1. Intent Classifier (ML) ──────► "email_automation"
         │
         ├──► 2. Node Recommender (ML) ───────► [formTrigger, gmail, slack]
         │
         ├──► 3. RAG Retriever (ChromaDB) ────► Top 3 similar n8n templates
         │
         ▼
    4. Generator (Google Gemini 2.0 Flash)  ◄── (Feeds context from layers 1-3)
         │
         ▼
    n8n Workflow JSON ───────► (Optional) Direct POST to remote n8n Instance
```

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend UI** | Streamlit |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib |
| **RAG / Embeddings** | ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **LLM Generator** | Google GenAI SDK (`gemini-2.0-flash`) |
| **Data Collection** | Requests, n8n Public API |
| **Testing** | Pytest, HTTPX |

## Setup & Installation

Follow these steps to run the complete pipeline locally:

**1. Clone the repository and navigate to the project directory:**
```bash
git clone <repository_url>
cd SME_WorkFlow_Final_Project
```

**2. Create and activate a virtual environment:**
*Windows:*
```bash
python -m venv .venv
.venv\Scripts\activate
```
*macOS/Linux:*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**5. Index templates for RAG search:**
```bash
python rag/embed_templates.py
```

**6. Start the FastAPI backend:**
```bash
uvicorn api.main:app --reload --port 8000
```

**7. Start the Streamlit UI (in a separate terminal):**
```bash
streamlit run app/streamlit_app.py
```

## Cloud Deployment

**1. Deploy Backend (FastAPI):**
Host the FastAPI app on Heroku, Render, or Railway using the included `Procfile`:
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```
Make sure to set `GEMINI_API_KEY` in your backend host's environment variables.

**2. Deploy Frontend (Streamlit Cloud):**
1. Connect your repository to [Streamlit Community Cloud](https://share.streamlit.io).
2. Set the Main File path to `app/streamlit_app.py`.
3. Click "Advanced Settings" ➔ "Secrets" and configure the backend URL (plus any n8n defaults if you want them pre-filled):
```toml
API_BASE_URL = "https://your-deployed-backend-url.com"

# Optional n8n defaults
N8N_URL = "https://your-n8n-instance.com"
N8N_API_KEY = "your-n8n-api-key"
```
4. Click Deploy!


## The Two ML Models

SME AutoFlow uses two traditional machine learning models strategically placed before the LLM to guide generation:
- **Intent Classifier:** A multi-class model (TF-IDF + `LogisticRegression` with `class_weight='balanced'`) that categorizes requests into one of 6 domains (like `data_sync` or `team_communication`). Achieves ~70% accuracy.
- **Node Recommender:** A multi-label model (TF-IDF + `OneVsRestClassifier(LogisticRegression)`) that predicts which of the top 30 n8n nodes should be present in the output. Evaluated using a Hamming Loss of ~0.11 and micro-F1 of ~0.54.

## Dataset

The ML models and RAG database are powered by 1,000 real-world, expert-created workflow templates downloaded natively from the [n8n public workflow registry](https://n8n.io/workflows/). This data was automatically labeled via `data/build_dataset.py`.

## Project Structure

```text
├── api/
│   └── main.py                     # FastAPI backend
├── app/
│   └── streamlit_app.py            # Streamlit UI
├── data/
│   ├── build_dataset.py            # Dataset processing & auto-labeling
│   ├── fetch_templates.py          # n8n template downloader
│   ├── labeled_dataset.csv         # Processed dataset for ML 
│   ├── processed_templates.json    # JSON data for RAG DB
│   └── raw_templates/              # 1,000 downloaded n8n JSONs
├── docs/                           # EDA and evaluation visualizations
├── generator/
│   └── gemini_generator.py         # Google Gemini integration script
├── models/
│   ├── intent_classifier/          # Trained model, encoder, train.py
│   └── node_recommender/           # Trained model, config, train.py
├── notebooks/                      # Jupyter notebooks (.ipynb & generators)
│   ├── 01_data_collection
│   ├── 02_eda
│   ├── 03_intent_classifier
│   └── 04_node_recommender
├── rag/
│   ├── embed_templates.py          # Vector embedding script
│   ├── retriever.py                # ChromaDB search capability
│   └── chroma_db/                  # Local vector database
├── tests/
│   ├── test_api.py                 # FastAPI integration mocks
│   └── test_classifier.py          # ML predictor unit tests
├── .env.example
├── .gitignore
└── requirements.txt
```

## Fellowship Context

This project was developed by Hammad Ali (FA23-BCS-007) as the Final Project for the GDGoC AI/ML Fellowship. It fulfills the core objectives of building an end-to-end AI application, integrating multiple systems including a complete machine learning data pipeline (collection, EDA, training), an internal RAG framework, an external LLM connection, and wrapping the solution in an accessible client-facing UI and an automated test suite.
