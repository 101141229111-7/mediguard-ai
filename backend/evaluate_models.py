"""
MediGuard AI - Model Evaluation Script
----------------------------------------
Computes accuracy, precision, recall, and F1 for:
  - standalone KNN
  - standalone Decision Tree
  - the hybrid ensemble (as saved in your .joblib bundle)
on a held-out test split, for both diabetes and heart datasets.

Run this from your backend/ folder, in the same activated venv:
    (venv) D:\\MediGuard-AI\\backend> python evaluate_models.py

It reuses the exact same preprocessing (imputer + scaler) already fitted
and saved inside your model bundles, so results reflect your real trained
models - nothing here is estimated or made up.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")


def evaluate_dataset(name, csv_filename, target_col="target"):
    bundle_path = os.path.join(MODELS_DIR, f"{name}_stack_model.joblib")
    bundle = joblib.load(bundle_path)
    feature_cols = bundle["feature_cols"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    ensemble_model = bundle["model"]

    csv_path = os.path.join(PROCESSED_DIR, csv_filename)
    df = pd.read_csv(csv_path)

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_imputed = imputer.transform(X_train)
    X_train_scaled = scaler.transform(X_train_imputed)

    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)

    ensemble_pred = ensemble_model.predict(X_test_scaled)

    def metrics(y_true, y_pred, label):
        return {
            "Model": label,
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    results = [
        metrics(y_test, knn_pred, "KNN (standalone)"),
        metrics(y_test, dt_pred, "Decision Tree (standalone)"),
        metrics(y_test, ensemble_pred, "Hybrid Ensemble (KNN + DT)"),
    ]

    print(f"\n===== {name.upper()} =====")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    # Adjust these filenames/target column names if yours differ
    evaluate_dataset("diabetes", "diabetes_processed.csv", target_col="Outcome")
    evaluate_dataset("heart", "heart_processed.csv", target_col="target")
