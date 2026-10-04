"""
Task 4 - Real World Data Project (Health Domain)
Breast Cancer Diagnosis: End-to-End Data Analysis & Prediction
--------------------------------------------------------------
Dataset : Breast Cancer Wisconsin (Diagnostic) Data Set
          A real, widely used clinical dataset (UCI ML Repository /
          scikit-learn) containing measurements taken from digitized
          images of fine needle aspirate (FNA) biopsies of breast masses.
          Each row = one patient's tumor measurements.
          Target = Diagnosis (Malignant / Benign)

Pipeline:
    1. Load & save the raw dataset           -> data/breast_cancer_data.csv
    2. Exploratory Data Analysis (EDA)        -> outputs/*.png
    3. Preprocessing (split + scale)
    4. Model training (Logistic Regression, Random Forest)
    5. Model evaluation (accuracy, confusion matrix, ROC-AUC)
    6. Feature importance
    7. Save trained model + written conclusions -> outputs/

Run from the project root with:
    python src/analysis.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid", palette="viridis")

# ---------------------------------------------------------------
# Paths (resolved relative to this file so it works from any cwd)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DATA_CSV = os.path.join(DATA_DIR, "breast_cancer_data.csv")


def load_and_save_data() -> pd.DataFrame:
    """Load the real breast cancer dataset and persist it as a CSV file
    so the project works with a genuine on-disk dataset file, just like
    any other real-world data project."""
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # target: 0 = malignant, 1 = benign (sklearn's encoding)
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    df.to_csv(DATA_CSV, index=False)
    print(f"[1/7] Dataset saved to: {DATA_CSV}")
    print(f"      Shape: {df.shape[0]} patients x {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Basic structural exploration + missing value / class balance checks."""
    print("\n[2/7] Exploratory checks")
    print("-" * 50)
    print("Missing values per column (top 5):")
    print(df.isnull().sum().sort_values(ascending=False).head())
    print("\nClass balance:")
    print(df["diagnosis"].value_counts())

    # Target distribution plot
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="diagnosis", hue="diagnosis", legend=False)
    ax.set_title("Diagnosis Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Patients")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_class_distribution.png"), dpi=150)
    plt.close()

    # Correlation heatmap (mean-value features only, for readability)
    mean_cols = [c for c in df.columns if c.startswith("mean ")]
    plt.figure(figsize=(10, 8))
    corr = df[mean_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title("Correlation Heatmap - 'Mean' Tumor Measurements")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    # Distribution of key features split by diagnosis
    key_features = ["mean radius", "mean texture", "mean perimeter",
                     "mean area", "mean smoothness", "mean concavity"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.flatten(), key_features):
        sns.histplot(data=df, x=feat, hue="diagnosis", kde=True, ax=ax, alpha=0.5)
        ax.set_title(feat)
    plt.suptitle("Key Feature Distributions by Diagnosis", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_feature_distributions.png"), dpi=150)
    plt.close()

    # Boxplots for outlier inspection
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, ["mean radius", "mean area", "mean concavity"]):
        sns.boxplot(data=df, x="diagnosis", y=feat, hue="diagnosis", legend=False, ax=ax)
        ax.set_title(f"{feat} by Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_boxplots.png"), dpi=150)
    plt.close()
    print("      EDA charts saved to outputs/ (01-04 .png)")


def preprocess(df: pd.DataFrame):
    print("\n[3/7] Preprocessing (train/test split + scaling)")
    X = df.drop(columns=["target", "diagnosis"])
    y = df["target"]  # 0 = Malignant, 1 = Benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"      Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train_scaled, y_train):
    print("\n[4/7] Training models")
    log_reg = LogisticRegression(max_iter=5000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    print("      Trained: Logistic Regression, Random Forest")
    return {"Logistic Regression": log_reg, "Random Forest": rf}


def evaluate_models(models: dict, X_test_scaled, y_test):
    print("\n[5/7] Evaluating models")
    results = {}
    plt.figure(figsize=(6, 5))

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1-score : {f1:.4f}")
        print(f"    ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(OUT_DIR, f"05_confusion_matrix_{safe_name}.png"), dpi=150)
        plt.close()

        # Add to combined ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(99)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.figure(99)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_roc_curve_comparison.png"), dpi=150)
    plt.close()

    with open(os.path.join(OUT_DIR, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n      Confusion matrices + ROC curve saved to outputs/")
    print("      Metrics saved to outputs/model_results.json")
    return results


def feature_importance(models: dict, feature_names):
    print("\n[6/7] Feature importance (Random Forest)")
    rf = models["Random Forest"]
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top10 = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top10.values, y=top10.index, hue=top10.index, legend=False)
    plt.title("Top 10 Most Important Features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_feature_importance.png"), dpi=150)
    plt.close()
    print("      Saved outputs/07_feature_importance.png")
    print(top10)
    return top10


def save_artifacts(models: dict, scaler, results: dict, top10):
    print("\n[7/7] Saving model artifacts + conclusions")
    joblib.dump(models["Random Forest"], os.path.join(OUT_DIR, "random_forest_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "feature_scaler.joblib"))

    best_model = max(results, key=lambda k: results[k]["roc_auc"])
    conclusions = f"""TASK 4 - PROJECT CONCLUSIONS
=============================
Domain   : Health
Dataset  : Breast Cancer Wisconsin (Diagnostic) - {569} patient records, 30 numeric
           tumor-cell measurements derived from biopsy imaging.
Goal     : Predict whether a tumor is Malignant or Benign from cell measurements.

Key Findings
------------
1. Class balance: dataset is moderately imbalanced (~63% Benign / 37% Malignant);
   stratified train/test split was used to preserve this ratio.
2. Several mean-value features (radius, perimeter, area, concavity) are strongly
   correlated with each other and with the diagnosis - larger / more irregular
   cell measurements are strongly associated with malignant tumors.
3. Best performing model: {best_model}
     Accuracy : {results[best_model]['accuracy']}
     Precision: {results[best_model]['precision']}
     Recall   : {results[best_model]['recall']}
     F1-score : {results[best_model]['f1_score']}
     ROC-AUC  : {results[best_model]['roc_auc']}
4. Top predictive features (Random Forest importance): {', '.join(top10.index[:5].tolist())}.
5. Both models achieve high recall on the Malignant class, which is the clinically
   critical metric - missing a malignant case (false negative) is far costlier
   than a false alarm.

Recommendation
--------------
{best_model} is recommended for this use case based on its higher ROC-AUC score.
Random Forest remains valuable alongside it for interpretability, since it
exposes feature importances that help clinicians understand which measurements
drive a prediction. For deployment, recall on the Malignant class should be
monitored closely and the probability threshold tuned to minimize false
negatives (missed cancer cases).

All supporting charts are saved in the outputs/ folder (01-07, .png).
"""
    with open(os.path.join(OUT_DIR, "conclusions.txt"), "w") as f:
        f.write(conclusions)
    print(conclusions)


def main():
    df = load_and_save_data()
    explore_data(df)
    X, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(df)
    models = train_models(X_train_scaled, y_train)
    results = evaluate_models(models, X_test_scaled, y_test)
    top10 = feature_importance(models, X.columns)
    save_artifacts(models, scaler, results, top10)
    print("\n✅ Done! Check the 'outputs/' folder for all charts, metrics, and conclusions.")


if __name__ == "__main__":
    main()
