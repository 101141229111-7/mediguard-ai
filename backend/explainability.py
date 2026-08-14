"""
MediGuard AI - Explainability + Model Persistence module
------------------------------------------------------------
Reuses the data prep + model training logic from model_training.py,
saves the trained stacking ensembles to disk with joblib (so the FastAPI
backend can load them without retraining), and generates SHAP
explanations showing which features drove individual predictions.

Requires: pip install shap --break-system-packages   (if not already installed)

Run from the backend/ folder, same venv as before:
    (venv) D:\\MediGuard-AI\\backend>python explainability.py
"""

import os
import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # so it works even without a display
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

from preprocessing_eda import (
    load_diabetes_real, load_diabetes_synthetic,
    load_heart_real, load_heart_synthetic,
    DIABETES_COLS, HEART_COLS, ZERO_AS_MISSING_COLS,
)
from model_training import prepare_dataset, real_only_split, RANDOM_STATE


def positive_class_shap(shap_values):
    """Normalize SHAP's output to a plain (samples, features) array of
    class-1 (positive/disease) contributions, regardless of SHAP version.
    Older SHAP: shap_values() returns a list of 2 arrays (one per class).
    Newer SHAP (0.45+): returns one 3D array shaped (samples, features, classes).
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed", "shap_plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_save_and_explain(dataset_name, real_df, syn_df, target_col, feature_cols, zero_cols=None):
    print(f"\n{'=' * 60}\n{dataset_name} - training + saving + SHAP\n{'=' * 60}")

    df, imputer = prepare_dataset(real_df, syn_df, target_col, feature_cols, zero_cols)
    X_train, X_test, y_train, y_test, scaler = real_only_split(df, target_col, feature_cols)

    stack = StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=7)),
            ("dt", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    )
    stack.fit(X_train, y_train)

    # --- Save the model + preprocessing so FastAPI can reuse them exactly,
    # without retraining and without any risk of preprocessing mismatch ---
    model_path = os.path.join(MODELS_DIR, f"{dataset_name.lower()}_stack_model.joblib")
    joblib.dump({
        "model": stack,
        "feature_cols": feature_cols,
        "imputer": imputer,
        "scaler": scaler,
    }, model_path)
    print(f"Saved model -> {model_path}")

    # --- SHAP explanation ---
    # Uses the Decision Tree component inside the ensemble: TreeExplainer
    # is fast and exact for tree models, and gives a concrete, defensible
    # feature-importance signal for the write-up.
    dt_component = stack.named_estimators_["dt"]
    explainer = shap.TreeExplainer(dt_component)
    shap_values = explainer.shap_values(X_test)
    shap_values_plot = positive_class_shap(shap_values)

    plt.figure()
    shap.summary_plot(shap_values_plot, X_test, feature_names=feature_cols, show=False)
    plot_path = os.path.join(PLOTS_DIR, f"{dataset_name.lower()}_shap_summary.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {plot_path}")

    # Explain one example patient from the test set, in plain feature terms
    sample_idx = 0
    sample_row = X_test[sample_idx:sample_idx + 1]
    sample_shap = positive_class_shap(explainer.shap_values(sample_row))

    contributions = pd.Series(sample_shap[0], index=feature_cols).sort_values(key=abs, ascending=False)
    predicted_class = stack.predict(sample_row)[0]
    print(f"\nExample explanation for one test patient (predicted class: {predicted_class}):")
    print(contributions.round(3).to_string())

    return stack


def main():
    diab_real = load_diabetes_real()
    diab_syn = load_diabetes_synthetic()
    diab_features = [c for c in DIABETES_COLS if c != "Outcome"]
    train_save_and_explain(
        "Diabetes", diab_real, diab_syn, "Outcome", diab_features,
        zero_cols=ZERO_AS_MISSING_COLS,
    )

    heart_real = load_heart_real()
    heart_syn = load_heart_synthetic()
    heart_features = [c for c in HEART_COLS if c != "target"]
    train_save_and_explain("Heart", heart_real, heart_syn, "target", heart_features)

    print(f"\n{'=' * 60}")
    print("DONE. Models saved to:", MODELS_DIR)
    print("SHAP plots saved to:", PLOTS_DIR)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()