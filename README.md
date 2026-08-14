# MediGuard AI

**Multi-Agent System for Chronic Disease Risk Monitoring and Early Warning**

MediGuard AI predicts the risk of chronic diseases (diabetes and heart disease) from patient health data, explains *why* a prediction was made using SHAP, and continuously monitors patient visit history to flag rising risk trends — all through a lightweight multi-agent architecture behind a single FastAPI backend.

## Problem Statement

Chronic diseases often show no early symptoms, and existing tools either give a bare risk score with no explanation, or only show past data without predicting or alerting on future risk. MediGuard AI addresses this with a system that predicts, explains, and continuously monitors disease risk.

## Features

- **Prediction agent** — hybrid KNN + Decision Tree ensemble estimates diabetes and heart disease risk from patient vitals
- **Explainability agent** — SHAP-based feature contributions accompany every prediction, so risk scores are never a black box
- **Monitoring/alerting agent** — tracks a patient's risk score across visits and flags consecutive-visit upward trends
- **Interactive dashboard** — Streamlit frontend for entering patient data, viewing risk badges, SHAP charts, and visit history trends

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| ML models | scikit-learn (KNN + Decision Tree stacked ensemble) |
| Explainability | SHAP |
| Frontend | Streamlit |
| Data | UCI Cleveland Heart Disease, Pima Indians Diabetes (real) + synthetic augmentation |

## Architecture

```
                ┌─────────────────────┐
Patient data →  │   FastAPI Backend    │
                │                      │
                │  ┌────────────────┐  │
                │  │ Prediction     │  │
                │  │ Agent (KNN+DT) │  │
                │  └────────────────┘  │
                │  ┌────────────────┐  │
                │  │ Explainability │  │──→ Streamlit Dashboard
                │  │ Agent (SHAP)   │  │
                │  └────────────────┘  │
                │  ┌────────────────┐  │
                │  │ Monitoring &   │  │
                │  │ Alert Agent    │  │
                │  └────────────────┘  │
                └─────────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check, lists loaded models |
| POST | `/predict/diabetes` | Diabetes risk prediction |
| POST | `/predict/heart` | Heart disease risk prediction |
| POST | `/explain/diabetes` | Diabetes prediction + SHAP feature contributions |
| POST | `/explain/heart` | Heart disease prediction + SHAP feature contributions |
| GET | `/monitor/{patient_id}` | Longitudinal visit history, risk trend, and alert status |

Interactive API docs available at `/docs` once the backend is running.

## Project Structure

```
MediGuard-AI/
├── backend/
│   ├── app.py                  # FastAPI app and routes
│   ├── explainability.py       # Model training + SHAP setup
│   ├── monitoring_agent.py     # Longitudinal monitoring logic
│   ├── requirements.txt
│   └── models/                 # Saved .joblib model bundles
├── frontend/
│   └── streamlit_app.py        # Streamlit dashboard
├── real/                       # Real UCI datasets
└── synthetic/                  # Synthetic augmentation + monitoring data
```

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

Backend runs at `http://127.0.0.1:8000` — visit `/docs` for the interactive API tester.

### Frontend

In a **second terminal**, with the backend still running:

```bash
cd frontend
venv\Scripts\activate
pip install streamlit requests pandas
streamlit run streamlit_app.py
```

Frontend opens automatically at `http://localhost:8501`.

> Both servers must run at the same time, in separate terminals, for the dashboard to work.

## Live Deployment

- Backend (FastAPI, hosted on Render): [https://mediguard-ai-sqas.onrender.com/docs](https://mediguard-ai-sqas.onrender.com/docs)
- Frontend (Streamlit, hosted on Streamlit Community Cloud): [https://mediguard-ai-shjwqi7hzhmplmuyrdafzu.streamlit.app](https://mediguard-ai-shjwqi7hzhmplmuyrdafzu.streamlit.app)

> Note: the backend runs on Render's free tier, which sleeps after inactivity. The first request after idle time may take 30–50 seconds to respond while it wakes up.

## Model Evaluation

Evaluated on a held-out 20% test split (stratified), using the same fitted preprocessing (imputer + scaler) as the deployed models.

**Diabetes**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN (standalone) | 0.7744 | 0.7152 | 0.5855 | 0.6439 |
| Decision Tree (standalone) | 0.7780 | 0.7160 | 0.6010 | 0.6535 |
| **Hybrid Ensemble (KNN + DT)** | **0.8195** | **0.8039** | **0.6373** | **0.7110** |

**Heart Disease**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN (standalone) | 0.7479 | 0.7095 | 0.7651 | 0.7362 |
| Decision Tree (standalone) | 0.7867 | 0.7514 | 0.8012 | 0.7755 |
| **Hybrid Ensemble (KNN + DT)** | **0.8837** | **0.8735** | **0.8735** | **0.8735** |

The hybrid ensemble outperforms both standalone models on every metric, for both datasets — most notably on heart disease, where it improves accuracy by ~10 points over either individual model.

## Future Scope

- Patient authentication and per-patient login
- Persistent database (PostgreSQL/MongoDB) for medical history instead of static CSVs
- Expansion to additional disease categories, each with dedicated models and datasets
- Wearable/IoT device integration for live vitals
- SMS/email alert notifications
- Docker containerization and cloud-native deployment
