"""
MediGuard-AI — Streamlit Frontend
----------------------------------
A simple UI on top of the FastAPI backend (app.py).

HOW TO RUN:
1. Make sure your FastAPI backend is running first:
       uvicorn app:app --reload
   (it should be live at http://127.0.0.1:8000)

2. In a NEW terminal (keep the backend one running), install streamlit if you
   haven't already:
       pip install streamlit requests pandas
   Then, from wherever you saved this file, run:
       streamlit run streamlit_app.py

3. It will open a browser tab automatically (usually http://localhost:8501)
"""

import streamlit as st
import requests
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
API_BASE = "https://mediguard-ai-sqas.onrender.com"

st.set_page_config(page_title="MediGuard-AI", page_icon="🩺", layout="wide")

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.title("🩺 MediGuard-AI")
page = st.sidebar.radio(
    "Choose a section",
    ["Diabetes", "Heart", "Monitoring"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend expected at:\n`{API_BASE}`")


def call_api(method: str, endpoint: str, payload: dict | None = None):
    """Small helper so every page handles connection errors the same way."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=payload, timeout=10)
        else:
            resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"Server returned {resp.status_code}: {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, (
            "Could not reach the backend. Is `uvicorn app:app --reload` "
            "still running in another terminal?"
        )
    except Exception as e:
        return None, str(e)


def contributions_chart(feature_contributions: dict):
    """Turn a {feature: value} dict into a horizontal bar chart, sorted by
    absolute impact, so the most influential features are easy to spot."""
    df = pd.DataFrame(
        {
            "feature": list(feature_contributions.keys()),
            "contribution": list(feature_contributions.values()),
        }
    )
    df["abs_contribution"] = df["contribution"].abs()
    df = df.sort_values("abs_contribution", ascending=True)
    st.bar_chart(df.set_index("feature")["contribution"])


def risk_badge(risk_score: float, alert: bool):
    if alert:
        st.error(f"⚠️ High Risk — score {risk_score:.3f}")
    else:
        st.success(f"✅ Lower Risk — score {risk_score:.3f}")


# ----------------------------------------------------------------------
# DIABETES PAGE
# ----------------------------------------------------------------------
if page == "Diabetes":
    st.header("Diabetes Risk Prediction")
    st.write("Enter patient values, then predict and/or explain the result.")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=148)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=35)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=125)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=33.6, format="%.1f")
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.627, format="%.3f"
        )
        age = st.number_input("Age", min_value=1, max_value=120, value=50)

    payload = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    btn_col1, btn_col2 = st.columns(2)
    predict_clicked = btn_col1.button("Predict", type="primary", key="diabetes_predict")
    explain_clicked = btn_col2.button("Predict + Explain", key="diabetes_explain")

    if predict_clicked:
        data, err = call_api("POST", "/predict/diabetes", payload)
        if err:
            st.error(err)
        else:
            risk_badge(data["risk_score"], data["alert"])
            st.caption(data.get("message", ""))

    if explain_clicked:
        data, err = call_api("POST", "/explain/diabetes", payload)
        if err:
            st.error(err)
        else:
            risk_badge(data["risk_score"], data["risk_score"] >= 0.6)
            st.subheader("What drove this prediction")
            contributions_chart(data["feature_contributions"])

# ----------------------------------------------------------------------
# HEART PAGE
# ----------------------------------------------------------------------
elif page == "Heart":
    st.header("Heart Disease Risk Prediction")
    st.write("Enter patient values, then predict and/or explain the result.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=63, key="h_age")
        sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1], key="h_sex")[0]
        cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=4, value=3, key="h_cp")
        trestbps = st.number_input("Resting BP (trestbps)", min_value=0, max_value=250, value=145, key="h_trestbps")
    with col2:
        chol = st.number_input("Cholesterol (chol)", min_value=0, max_value=700, value=233, key="h_chol")
        fbs = st.selectbox("Fasting Blood Sugar > 120 (fbs)", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1], key="h_fbs")[0]
        restecg = st.number_input("Resting ECG (restecg)", min_value=0, max_value=2, value=0, key="h_restecg")
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=0, max_value=250, value=150, key="h_thalach")
    with col3:
        exang = st.selectbox("Exercise Angina (exang)", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1], key="h_exang")[0]
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=2.3, format="%.1f", key="h_oldpeak")
        slope = st.number_input("Slope", min_value=0, max_value=3, value=0, key="h_slope")
        ca = st.number_input("Major Vessels (ca)", min_value=0, max_value=4, value=0, key="h_ca")
        thal = st.number_input("Thal", min_value=0, max_value=7, value=1, key="h_thal")

    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }

    btn_col1, btn_col2 = st.columns(2)
    predict_clicked = btn_col1.button("Predict", type="primary", key="heart_predict")
    explain_clicked = btn_col2.button("Predict + Explain", key="heart_explain")

    if predict_clicked:
        data, err = call_api("POST", "/predict/heart", payload)
        if err:
            st.error(err)
        else:
            risk_badge(data["risk_score"], data["alert"])
            st.caption(data.get("message", ""))

    if explain_clicked:
        data, err = call_api("POST", "/explain/heart", payload)
        if err:
            st.error(err)
        else:
            risk_badge(data["risk_score"], data["risk_score"] >= 0.6)
            st.subheader("What drove this prediction")
            contributions_chart(data["feature_contributions"])

# ----------------------------------------------------------------------
# MONITORING PAGE
# ----------------------------------------------------------------------
elif page == "Monitoring":
    st.header("Longitudinal Patient Monitoring")
    st.write("Look up a patient's visit history and risk trend over time.")

    patient_id = st.text_input("Patient ID", value="M00003")
    lookup_clicked = st.button("Look up patient", type="primary")

    if lookup_clicked:
        data, err = call_api("GET", f"/monitor/{patient_id}")
        if err:
            st.error(err)
        else:
            st.subheader(f"Patient {data['patient_id']} — {data['num_visits']} visits")

            top1, top2, top3 = st.columns(3)
            top1.metric("Latest Risk Score", f"{data['latest_risk_score']:.3f}")
            top2.metric("Trend", data["trend"])
            top3.metric("Alert", "⚠️ Yes" if data["alert"] else "✅ No")

            if data["alert"]:
                st.warning(data.get("alert_reason", "Alert triggered."))

            visits_df = pd.DataFrame(data["visit_history"])
            st.subheader("Risk score across visits")
            st.line_chart(visits_df.set_index("visit_number")["computed_risk"])

            st.subheader("Full visit history")
            st.dataframe(visits_df, use_container_width=True)
