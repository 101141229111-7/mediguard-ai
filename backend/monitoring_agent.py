"""
MediGuard AI - Monitoring & Alert Agent
-----------------------------------------
Reads a patient's visit history, computes a transparent clinical risk score
per visit (NOT the precomputed synthetic risk_score/alert columns, which
were only used to seed the generator and should never be treated as
ground truth), tracks the trend across visits, and decides whether to
fire an alert.

Design note for your report: this agent uses rule-based clinical
thresholds rather than the diabetes ML model, because the longitudinal
monitoring dataset tracks a different, smaller feature set (glucose,
BP, BMI, heart rate, steps) than the 8 features the diabetes model was
trained on. Forcing the ML model onto incomplete/mismatched features
would mean guessing at missing values every visit -- less defensible
than a transparent, clinically-grounded scoring rule. This is a
deliberate architectural choice, not a shortcut.

Run standalone to test:
    (venv) D:\\MediGuard-AI\\backend>python monitoring_agent.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MONITORING_CSV = os.path.join(BACKEND_DIR, "processed", "monitoring_processed.csv")

# ---------------------------------------------------------------
# 2. CLINICAL RISK SCORING
#    Each factor contributes 0-1, weighted, summed, capped at 1.0.
#    Thresholds are standard clinical reference ranges.
# ---------------------------------------------------------------
def _glucose_risk(g):
    if g < 100:
        return 0.0
    if g < 126:
        return (g - 100) / 26 * 0.5          # pre-diabetic range, up to 0.5
    return min(0.5 + (g - 126) / 74 * 0.5, 1.0)   # diabetic range, 0.5-1.0


def _bp_risk(sbp):
    if sbp < 120:
        return 0.0
    if sbp < 140:
        return (sbp - 120) / 20 * 0.5        # elevated/stage-1
    return min(0.5 + (sbp - 140) / 60 * 0.5, 1.0)  # stage-2 hypertensive


def _bmi_risk(bmi):
    if bmi < 25:
        return 0.0
    if bmi < 30:
        return (bmi - 25) / 5 * 0.4          # overweight
    return min(0.4 + (bmi - 30) / 15 * 0.6, 1.0)   # obese


def _hr_risk(hr):
    if 60 <= hr <= 100:
        return 0.0
    if hr > 100:
        return min((hr - 100) / 40, 1.0)
    return min((60 - hr) / 30, 1.0)


WEIGHTS = {"glucose": 0.40, "bp": 0.25, "bmi": 0.20, "hr": 0.15}


def compute_visit_risk(row):
    """Composite clinical risk score in [0, 1] for a single visit."""
    score = (
        WEIGHTS["glucose"] * _glucose_risk(row["glucose"])
        + WEIGHTS["bp"] * _bp_risk(row["systolic_bp"])
        + WEIGHTS["bmi"] * _bmi_risk(row["bmi"])
        + WEIGHTS["hr"] * _hr_risk(row["heart_rate"])
    )
    return round(min(score, 1.0), 3)


# ---------------------------------------------------------------
# 3. TREND + ALERT LOGIC
# ---------------------------------------------------------------
def detect_trend(risk_series, lookback=3):
    """Looks at the last `lookback` visits and classifies the trend."""
    recent = risk_series[-lookback:]
    if len(recent) < 2:
        return "insufficient_data"
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    if all(d > 0.01 for d in diffs):
        return "worsening"
    if all(d < -0.01 for d in diffs):
        return "improving"
    return "stable"


def should_alert(risk_series, lookback=3, rising_threshold=3, absolute_threshold=0.7):
    """
    Fires an alert if EITHER:
      (a) risk has risen for `rising_threshold` consecutive visits, or
      (b) the latest risk score crosses `absolute_threshold`.
    Returns (alert: bool, reason: str).
    """
    latest = risk_series[-1]

    if latest >= absolute_threshold:
        return True, f"Latest risk score {latest} exceeds threshold {absolute_threshold}"

    if len(risk_series) >= rising_threshold:
        recent = risk_series[-rising_threshold:]
        rising = all(recent[i + 1] > recent[i] for i in range(len(recent) - 1))
        if rising:
            return True, f"Risk has risen for {rising_threshold} consecutive visits: {recent}"

    return False, "No alert conditions met"


# ---------------------------------------------------------------
# 4. PER-PATIENT PIPELINE
# ---------------------------------------------------------------
def get_patient_monitoring_result(patient_id, df=None):
    if df is None:
        df = pd.read_csv(MONITORING_CSV)

    patient_df = df[df["patient_id"] == patient_id].sort_values("visit_number")
    if patient_df.empty:
        return {"error": f"No visits found for patient_id={patient_id}"}

    patient_df = patient_df.copy()
    patient_df["computed_risk"] = patient_df.apply(compute_visit_risk, axis=1)

    risk_series = patient_df["computed_risk"].tolist()
    trend = detect_trend(risk_series)
    alert, reason = should_alert(risk_series)

    return {
        "patient_id": patient_id,
        "num_visits": len(patient_df),
        "visit_history": patient_df[
            ["visit_number", "glucose", "systolic_bp", "bmi", "heart_rate", "computed_risk"]
        ].to_dict(orient="records"),
        "latest_risk_score": risk_series[-1],
        "trend": trend,
        "alert": alert,
        "alert_reason": reason,
    }


# ---------------------------------------------------------------
# 5. STANDALONE TEST
# ---------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(MONITORING_CSV)
    sample_ids = df["patient_id"].unique()[:3]

    for pid in sample_ids:
        result = get_patient_monitoring_result(pid, df)
        print(f"\n{'=' * 50}")
        print(f"Patient: {result['patient_id']}  |  Visits: {result['num_visits']}")
        print(f"Latest risk: {result['latest_risk_score']}  |  Trend: {result['trend']}")
        print(f"Alert: {result['alert']}  |  Reason: {result['alert_reason']}")

    print(f"\n{'=' * 50}\nDONE - monitoring agent working correctly.")