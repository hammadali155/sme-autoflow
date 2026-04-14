# SME AutoFlow — Agent Context

## What This Project Is
SME AutoFlow is an AI-powered n8n automation workflow generator built as a solo final project for the GDGoC AI/ML Fellowship (April 2026). The system takes a plain-language business problem from a user, classifies it into one of 6 automation categories using a trained ML model, recommends relevant n8n nodes using a second trained model, retrieves similar workflows from a vector database using RAG, and then uses the Gemini API to generate a valid, ready-to-import n8n workflow JSON. The final output is auto-deployed to the user's n8n instance via REST API — no technical knowledge required from the user.

## Developer
- **Name:** Hammad Ali
- **ID:** FA23-BCS-007
- **University:** COMSATS University Islamabad, Attock Campus
- **Program:** BS Computer Science (Semester 6, Session 2023–2027)
- **Fellowship:** GDGoC AI/ML Fellowship

## Project Goal
The project must satisfy fellowship evaluation requirements:
- Two independently trained and saved ML models (`.pkl` files)
- Real dataset (scraped from n8n public API + custom-labelled)
- Evaluation metrics: Accuracy, F1-score, Precision, Recall, Hamming Loss
- A working deployed application (Streamlit Cloud)
- GitHub repository with all code, notebooks, and saved models

## Tech Stack
| Layer | Technology |
|---|---|
| ML Models | Scikit-learn (TF-IDF + Random Forest, TF-IDF + OneVsRest Logistic Regression) |
| Vector DB | ChromaDB (persistent) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| AI Generation | Gemini 2.0 Flash API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |
| Language | Python 3.11+ |

## The Two Trained Models
1. **Intent Classifier** (`models/intent_classifier/intent_model.pkl`)
   - Input: Free-text business description
   - Output: One of 6 categories → `communication`, `alerts`, `data_sync`, `reporting`, `scheduling`, `ecommerce`
   - Algorithm: TF-IDF vectoriser + Logistic Regression pipeline

2. **Node Recommender** (`models/node_recommender/node_model.pkl`)
   - Input: Free-text business description
   - Output: List of predicted n8n nodes (multi-label)
   - Algorithm: TF-IDF vectoriser + OneVsRest Logistic Regression

## Directory Structure
```
sme-autoflow/
├── data/
│   ├── raw_templates/
│   ├── processed_templates.json
│   └── labeled_dataset.csv
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_intent_classifier.ipynb
│   └── 04_node_recommender.ipynb
├── models/
│   ├── intent_classifier/
│   │   ├── train.py
│   │   ├── intent_model.pkl
│   │   └── label_encoder.pkl
│   └── node_recommender/
│       ├── train.py
│       ├── node_model.pkl
│       └── mlb.pkl
├── rag/
│   ├── embed_templates.py
│   ├── retriever.py
│   └── chroma_db/          ← auto-generated, in .gitignore
├── generator/
│   └── gemini_generator.py
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── tests/
├── docs/
│   └── SRS_SME_AutoFlow.pdf
├── .env                    ← never commit this
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment Variables (in .env)
```
GEMINI_API_KEY=your_key_here
N8N_URL=https://your-n8n-instance.com
N8N_API_KEY=your_n8n_key_here
```

## Key Rules for the Agent
- **Never commit `.env`** — always check `.gitignore` before pushing
- **Never commit `chroma_db/`** — it is large and auto-generated; always in `.gitignore`
- **Always save both `.pkl` files** after training — evaluators check for them
- **All notebook outputs must be visible** — run all cells before saving notebooks
- **Use `--break-system-packages`** flag whenever installing pip packages in this environment
- **Streamlit app must not require n8n credentials to browse** — make n8n fields optional with a preview mode
- **FastAPI and Streamlit run on different ports** — FastAPI on `8000`, Streamlit on `8501`
- **Gemini model to use:** `gemini-2.0-flash` (not pro — stay within free tier)
- **All API calls wrapped in try/except** — never let a bad Gemini response crash the app

## Data Source
n8n public template API: `https://api.n8n.io/api/templates/search?page=1&rows=50`
Fetch pages 1–20 to get ~1000 templates. Each template has `name`, `description`, `nodes`, and `tags`.

## Coding Standards
- Python type hints on all functions
- Docstrings on all classes and public methods
- `requirements.txt` must be kept up to date
- No hardcoded API keys anywhere in source files
- Use `python-dotenv` to load `.env`