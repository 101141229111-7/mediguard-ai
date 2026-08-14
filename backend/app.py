"""
MediGuard AI - FastAPI backend
--------------------------------
Loads the saved diabetes and heart models (each bundled with its fitted
imputer + scaler from explainability.py) and exposes them as four
logical agents behind REST endpoints:
  - ingestion   : implicit, via the request body itself
  - prediction  : POST /predict/{diabetes|heart}
  - explainability : POST /explain/{diabetes|heart}
  - monitoring/alerting : GET /monitor/{patient_id}

Requires: pip install fastapi "uvicorn[standard]" --break-system-packages
          (if not already installed)

Run from the backend/ folder, same venv as before:
    (venv) D:\\MediGuard-AI\\backend>python -m uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive tester.
"""

import os
import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explainability import positive_class_shap
from monitoring_agent import get_patient_monitoring_result

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Risk score threshold above which an alert fires. Tune this based on
# your project's tolerance for false alarms vs missed risk.
ALERT_THRESHOLD = 0.6

# Define the app FIRST, before anything that could fail (like loading the
# model files). This way, even if bundle-loading below throws an error,
# uvicorn still finds a valid "app" object and shows you the REAL
# underlying exception instead of a confusing "Attribute app not found".
app = FastAPI(title="MediGuard AI API")


# ---------------------------------------------------------------
# 1. LOAD SAVED MODEL BUNDLES ONCE AT STARTUP
# ---------------------------------------------------------------
def load_bundle(name):
    path = os.path.join(MODELS_DIR, f"{name}_stack_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run explainability.py first to train and save the models."
        )
    bundle = joblib.load(path)
    bundle["explainer"] = shap.TreeExplainer(bundle["model"].named_estimators_["dt"])
    return bundle


print("Loading model bundles...")
BUNDLES = {
    "diabetes": load_bundle("diabetes"),
    "heart": load_bundle("heart"),
}
print(f"Loaded bundles for: {list(BUNDLES.keys())}")


# ---------------------------------------------------------------
# 2. REQUEST SCHEMAS - field order matches feature_cols exactly
# ---------------------------------------------------------------
class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., gt=0, description="0 is treated as missing during training; send a real reading")
    BloodPressure: float = Field(..., gt=0)
    SkinThickness: float = Field(..., gt=0)
    Insulin: float = Field(..., gt=0)
    BMI: float = Field(..., gt=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., gt=0)


class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


# ---------------------------------------------------------------
# 3. SHARED PREDICTION + EXPLANATION LOGIC
# ---------------------------------------------------------------
def run_prediction(dataset_name, input_dict):
    bundle = BUNDLES[dataset_name]
    feature_cols = bundle["feature_cols"]

    # Preserve exact column order the model was trained on
    raw = np.array([[input_dict[col] for col in feature_cols]], dtype=float)

    imputed = bundle["imputer"].transform(raw)
    scaled = bundle["scaler"].transform(imputed)

    model = bundle["model"]
    pred_class = int(model.predict(scaled)[0])
    risk_score = float(model.predict_proba(scaled)[0][1])

    return scaled, pred_class, risk_score


def run_explanation(dataset_name, scaled_row):
    bundle = BUNDLES[dataset_name]
    feature_cols = bundle["feature_cols"]
    shap_vals = positive_class_shap(bundle["explainer"].shap_values(scaled_row))[0]
    return {
        col: round(float(val), 4)
        for col, val in sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))
    }


def alert_for(risk_score):
    if risk_score >= ALERT_THRESHOLD:
        return {"alert": True, "message": f"Risk score {risk_score:.2f} exceeds threshold ({ALERT_THRESHOLD}) - flagged for review."}
    return {"alert": False, "message": "Risk score within normal range."}


# ---------------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "models_loaded": list(BUNDLES.keys())}


@app.post("/predict/diabetes")
def predict_diabetes(payload: DiabetesInput):
    try:
        scaled, pred_class, risk_score = run_prediction("diabetes", payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "predicted_class": pred_class,
        "risk_score": round(risk_score, 4),
        **alert_for(risk_score),
    }


@app.post("/predict/heart")
def predict_heart(payload: HeartInput):
    try:
        scaled, pred_class, risk_score = run_prediction("heart", payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "predicted_class": pred_class,
        "risk_score": round(risk_score, 4),
        **alert_for(risk_score),
    }


@app.post("/explain/diabetes")
def explain_diabetes(payload: DiabetesInput):
    try:
        scaled, pred_class, risk_score = run_prediction("diabetes", payload.model_dump())
        contributions = run_explanation("diabetes", scaled)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "predicted_class": pred_class,
        "risk_score": round(risk_score, 4),
        "feature_contributions": contributions,
    }


@app.post("/explain/heart")
def explain_heart(payload: HeartInput):
    try:
        scaled, pred_class, risk_score = run_prediction("heart", payload.model_dump())
        contributions = run_explanation("heart", scaled)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "predicted_class": pred_class,
        "risk_score": round(risk_score, 4),
        "feature_contributions": contributions,
    }


@app.get("/monitor/{patient_id}")
def monitor_patient(patient_id: str):
    result = get_patient_monitoring_result(patient_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result