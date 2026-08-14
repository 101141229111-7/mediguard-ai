"""
MediGuard AI - Model Training module
-------------------------------------
Loads real+synthetic diabetes and heart disease data (reusing the cleaning
and rebalancing logic from preprocessing_eda.py), applies proper KNN-based
imputation, scales features, splits off a REAL-DATA-ONLY test set, trains
standalone KNN and Decision Tree baselines, then builds a stacking
ensemble (KNN + Decision Tree feeding a Logistic Regression meta-learner)
and compares all three on the same test set.

Requires: pip install scikit-learn --break-system-packages   (if not already installed)

Run from the backend/ folder, same venv as before:
    (venv) D:\\MediGuard-AI\\backend>python model_training.py
"""

import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Reuses the loaders + fixes you already built and verified in preprocessing_eda.py
from preprocessing_eda import (
    load_diabetes_real, load_diabetes_synthetic,
    load_heart_real, load_heart_synthetic,
    fix_zero_as_missing, rebalance_synthetic_to_real,
    DIABETES_COLS, HEART_COLS, ZERO_AS_MISSING_COLS,
)

RANDOM_STATE = 42


# ---------------------------------------------------------------
# 1. DATA PREP
# ---------------------------------------------------------------
def prepare_dataset(real_df, syn_df, target_col, feature_cols, zero_cols=None):
    """Rebalance synthetic to match real class distribution, fix
    zero-as-missing where relevant, then impute with KNNImputer (uses
    correlated features per row instead of one flat median for everyone).
    Returns the cleaned dataframe AND the fitted imputer, so the same
    imputer can be reused later on new incoming patient data."""
    syn_df = rebalance_synthetic_to_real(real_df, syn_df, target_col)
    df = pd.concat([real_df, syn_df], ignore_index=True)

    if zero_cols:
        df = fix_zero_as_missing(df, zero_cols)

    imputer = KNNImputer(n_neighbors=5)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    return df, imputer


def real_only_split(df, target_col, feature_cols, test_size=0.2):
    """Test set drawn ONLY from real rows, stratified by target, so the
    reported metrics reflect real-world generalization rather than the
    synthetic generator's own patterns. Training set = remaining real rows
    + all synthetic rows. Returns the fitted scaler too, so new incoming
    patient data can be scaled identically at prediction time."""
    real_df = df[df["source"] == "real"]
    train_real, test_real = train_test_split(
        real_df, test_size=test_size, stratify=real_df[target_col],
        random_state=RANDOM_STATE,
    )
    train_df = pd.concat([train_real, df[df["source"] == "synthetic"]], ignore_index=True)

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_real[feature_cols], test_real[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------
# 2. EVALUATION
# ---------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 3),
        "Precision": round(precision_score(y_test, preds), 3),
        "Recall": round(recall_score(y_test, preds), 3),
        "F1": round(f1_score(y_test, preds), 3),
        "AUC": round(roc_auc_score(y_test, probs), 3),
    }


# ---------------------------------------------------------------
# 3. PIPELINE PER DATASET
# ---------------------------------------------------------------
def run_pipeline(dataset_name, real_df, syn_df, target_col, feature_cols, zero_cols=None):
    print(f"\n{'=' * 60}\n{dataset_name}\n{'=' * 60}")

    df, imputer = prepare_dataset(real_df, syn_df, target_col, feature_cols, zero_cols)
    X_train, X_test, y_train, y_test, scaler = real_only_split(df, target_col, feature_cols)

    knn = KNeighborsClassifier(n_neighbors=7)
    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    knn.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Stacking: KNN + Decision Tree predictions feed a Logistic Regression
    # meta-learner, which learns how much to trust each base model.
    stack = StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=7)),
            ("dt", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    )
    stack.fit(X_train, y_train)

    results = [
        evaluate("KNN (baseline)", knn, X_test, y_test),
        evaluate("Decision Tree (baseline)", dt, X_test, y_test),
        evaluate("Hybrid Stacking Ensemble", stack, X_test, y_test),
    ]

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df, stack


# ---------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
    os.makedirs(out_dir, exist_ok=True)

    diab_real = load_diabetes_real()
    diab_syn = load_diabetes_synthetic()
    diab_features = [c for c in DIABETES_COLS if c != "Outcome"]
    diab_results, diab_model = run_pipeline(
        "Diabetes", diab_real, diab_syn, "Outcome", diab_features,
        zero_cols=ZERO_AS_MISSING_COLS,
    )
    diab_results.to_csv(os.path.join(out_dir, "diabetes_model_results.csv"), index=False)

    heart_real = load_heart_real()
    heart_syn = load_heart_synthetic()
    heart_features = [c for c in HEART_COLS if c != "target"]
    heart_results, heart_model = run_pipeline(
        "Heart Disease", heart_real, heart_syn, "target", heart_features,
    )
    heart_results.to_csv(os.path.join(out_dir, "heart_model_results.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("DONE. Model comparison tables saved to:", out_dir)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()