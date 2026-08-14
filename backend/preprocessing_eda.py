"""
MediGuard AI - Preprocessing + EDA module
-------------------------------------------
Loads all 5 datasets (2 real, 3 synthetic), standardizes column names so
real and synthetic files share identical schemas, runs a data-quality
check, and saves EDA plots + a cleaned version of each file.

Folder assumptions (adjust DATA_DIR below if your layout differs):
    MediGuard-AI/
    ├── real/
    │   ├── pima_diabetes_real.csv
    │   └── cleveland_heart_real.csv
    ├── synthetic/
    │   ├── mediguard_diabetes_synthetic.csv
    │   ├── mediguard_heart_synthetic.csv
    │   └── mediguard_longitudinal_monitoring_synthetic.csv
    └── backend/
        ├── preprocessing_eda.py   <- this file
        └── processed/             <- outputs land here
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works even without a display
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------
# 1. PATHS  -- edit these three lines if your folder layout differs
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_REAL = os.path.join(PROJECT_ROOT, "real")
DATA_SYNTHETIC = os.path.join(PROJECT_ROOT, "synthetic")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
PLOTS_DIR = os.path.join(OUT_DIR, "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DIABETES_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Columns where 0 is not a physiologically valid reading -- in the Pima
# dataset (and this synthetic mirror of it) 0 means "not recorded", not
# "measured as zero". Pregnancies is legitimately allowed to be 0, so it's
# excluded here on purpose.
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ---------------------------------------------------------------
# 2. LOADERS  -- standardize real + synthetic to the SAME column names
# ---------------------------------------------------------------
def load_diabetes_real():
    df = pd.read_csv(os.path.join(DATA_REAL, "pima_diabetes_real.csv"))
    df["source"] = "real"
    return df[DIABETES_COLS + ["source"]]


def load_diabetes_synthetic():
    df = pd.read_csv(os.path.join(DATA_SYNTHETIC, "mediguard_diabetes_synthetic.csv"))
    rename_map = {
        "pregnancies": "Pregnancies", "glucose": "Glucose",
        "blood_pressure": "BloodPressure", "skin_thickness": "SkinThickness",
        "insulin": "Insulin", "bmi": "BMI",
        "diabetes_pedigree": "DiabetesPedigreeFunction", "age": "Age",
        "outcome": "Outcome",
    }
    df = df.rename(columns=rename_map)
    df["source"] = "synthetic"
    keep = [c for c in DIABETES_COLS if c in df.columns]
    return df[keep + ["source"]]


def load_heart_real():
    df = pd.read_csv(os.path.join(DATA_REAL, "cleveland_heart_real.csv"))
    # Cleveland target is 0-4 severity; binarize to match standard practice
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)
    df["source"] = "real"
    return df[HEART_COLS + ["source"]]


def load_heart_synthetic():
    df = pd.read_csv(os.path.join(DATA_SYNTHETIC, "mediguard_heart_synthetic.csv"))
    keep = [c for c in HEART_COLS if c in df.columns]
    df = df[keep].copy()
    df["source"] = "synthetic"
    return df


def load_monitoring():
    return pd.read_csv(os.path.join(DATA_SYNTHETIC, "mediguard_longitudinal_monitoring_synthetic.csv"))


# ---------------------------------------------------------------
# 3. SYNTHETIC CLASS-BALANCE FIX
# ---------------------------------------------------------------
def rebalance_synthetic_to_real(real_df, syn_df, target_col, random_state=42):
    """Resample the synthetic data so its class distribution matches the
    real data's class distribution. Keeps the synthetic set roughly the
    same total size, but re-mixes which rows are included so the combined
    real+synthetic dataset doesn't inherit a label skew the generator
    introduced (e.g. synthetic rows being disproportionately positive)."""
    real_props = real_df[target_col].value_counts(normalize=True)
    n_total = len(syn_df)
    print(f"  Real-data {target_col} distribution used as target: "
          f"{real_props.round(3).to_dict()}")

    resampled = []
    for cls, prop in real_props.items():
        cls_df = syn_df[syn_df[target_col] == cls]
        n_target = int(round(n_total * prop))
        if len(cls_df) == 0:
            print(f"  WARNING: synthetic data has zero rows for {target_col}={cls}, skipping")
            continue
        replace = n_target > len(cls_df)
        resampled.append(cls_df.sample(n=n_target, replace=replace, random_state=random_state))

    out = pd.concat(resampled, ignore_index=True)
    print(f"  Synthetic rows: {len(syn_df)} -> {len(out)} after rebalancing")
    return out


# ---------------------------------------------------------------
# 4. ZERO-AS-MISSING FIX (diabetes only)
# ---------------------------------------------------------------
def fix_zero_as_missing(df, cols):
    """Convert physiologically-impossible 0 readings to NaN so they get
    counted as missing and properly imputed, instead of silently dragging
    down means/medians."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            n_zero = (df[col] == 0).sum()
            if n_zero:
                print(f"  -> {col}: converting {n_zero} zero-readings to NaN")
            df[col] = df[col].replace(0, np.nan)
    return df


# ---------------------------------------------------------------
# 4. DATA QUALITY REPORT
# ---------------------------------------------------------------
def quality_report(name, df, target_col=None):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Shape: {df.shape}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"Missing values:\n{missing}")
    else:
        print("Missing values: none")
    if target_col and target_col in df.columns:
        print(f"\nClass balance ({target_col}):")
        print(df[target_col].value_counts(normalize=True).round(3))
        if "source" in df.columns:
            print(f"\nClass balance by source (real vs synthetic):")
            print(df.groupby("source")[target_col].value_counts(normalize=True).round(3))
    print(f"\nSummary stats:\n{df.describe().round(2).T}")


# ---------------------------------------------------------------
# 5. EDA PLOTS
# ---------------------------------------------------------------
def save_eda_plots(name, df, target_col, feature_cols):
    sns.set_theme(style="darkgrid")

    # class balance
    plt.figure(figsize=(5, 4))
    df[target_col].value_counts().sort_index().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title(f"{name} - Class Balance ({target_col})")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_class_balance.png"), dpi=120)
    plt.close()

    # correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"{name} - Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_correlation.png"), dpi=120)
    plt.close()

    # distribution of first 4 numeric features
    numeric_feats = df[feature_cols].select_dtypes(include="number").columns[:4]
    fig, axes = plt.subplots(1, len(numeric_feats), figsize=(4 * len(numeric_feats), 3.5))
    if len(numeric_feats) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_feats):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{name}_distributions.png"), dpi=120)
    plt.close()

    print(f"Saved plots for {name} -> {PLOTS_DIR}")


# ---------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------
def main():
    # --- Diabetes ---
    diab_real = load_diabetes_real()
    diab_syn = load_diabetes_synthetic()

    print("\nRebalancing synthetic diabetes data to match real class distribution...")
    diab_syn = rebalance_synthetic_to_real(diab_real, diab_syn, "Outcome")

    diab_all = pd.concat([diab_real, diab_syn], ignore_index=True)

    print("\nFixing zero-as-missing values in diabetes data...")
    diab_all = fix_zero_as_missing(diab_all, ZERO_AS_MISSING_COLS)

    quality_report("Diabetes (real + synthetic combined)", diab_all, target_col="Outcome")

    diab_features = [c for c in DIABETES_COLS if c != "Outcome"]
    save_eda_plots("diabetes", diab_all, "Outcome", diab_features)

    diab_clean = diab_all.copy()
    for col in diab_features:
        diab_clean[col] = diab_clean[col].fillna(diab_clean[col].median())
    diab_clean.to_csv(os.path.join(OUT_DIR, "diabetes_processed.csv"), index=False)

    # --- Heart ---
    heart_real = load_heart_real()
    heart_syn = load_heart_synthetic()

    print("\nRebalancing synthetic heart data to match real class distribution...")
    heart_syn = rebalance_synthetic_to_real(heart_real, heart_syn, "target")

    heart_all = pd.concat([heart_real, heart_syn], ignore_index=True)
    quality_report("Heart Disease (real + synthetic combined)", heart_all, target_col="target")

    heart_features = [c for c in HEART_COLS if c != "target"]
    save_eda_plots("heart", heart_all, "target", heart_features)

    heart_clean = heart_all.copy()
    for col in heart_features:
        heart_clean[col] = heart_clean[col].fillna(heart_clean[col].median())
    heart_clean.to_csv(os.path.join(OUT_DIR, "heart_processed.csv"), index=False)

    # --- Monitoring ---
    monitoring = load_monitoring()
    quality_report("Longitudinal Monitoring", monitoring)
    monitoring.to_csv(os.path.join(OUT_DIR, "monitoring_processed.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("DONE. Processed files saved to:", OUT_DIR)
    print("EDA plots saved to:", PLOTS_DIR)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()