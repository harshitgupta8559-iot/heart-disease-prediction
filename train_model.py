"""
Machine Learning Model Training and Evaluation Script
=====================================================
Project: Heart Disease Prediction System (Educational Demonstration)

DISCLAIMER:
This project and its trained models are strictly for educational and demonstration
purposes. It is NOT a medical diagnostic tool and must NOT be used for real-life
clinical decision-making.

What is Training Data?
----------------------
Training data (X_train, y_train: 80% / 241 patients) is the textbook that the algorithms
study to discover relationships between clinical features (such as blood pressure,
cholesterol, and ECG results) and the presence of heart disease.

What is Test Data?
------------------
Test data (X_test, y_test: 20% / 61 patients) is the final exam. These patients are
completely hidden from the algorithms during training so we can honestly measure how
well each model performs on never-before-seen data.

Models Evaluated:
-----------------
1. Logistic Regression: Uses a mathematical sigmoid function to assign weights to
   each medical feature and predict the probability of heart disease. Requires feature
   scaling (StandardScaler) because blood pressure and cholesterol numbers are much
   larger than age or ST-depression values.
2. Decision Tree Classifier: Builds a hierarchy of rule-based questions (e.g., 'Is
   chest pain > 0?', 'Is max heart rate > 150?'). Highly interpretable.
3. Random Forest Classifier: An ensemble of 100 individual decision trees voting together.
   Combines predictions from multiple trees to reduce individual errors and variance.

Evaluation Metrics Explained:
-----------------------------
- Accuracy:  (Correct Predictions) / (Total Predictions) - Overall correctness.
- Precision: (True Positives) / (True Positives + False Positives) - When the model
             predicts heart disease, how often is it right?
- Recall:    (True Positives) / (True Positives + False Negatives) - How many of the
             actual heart disease patients did the model catch? (Crucial in healthcare).
- F1-Score:  Harmonic mean of Precision and Recall. Balances both metrics into a single score.
- Confusion Matrix: A 2x2 breakdown:
    [True Negatives (TN)   False Positives (FP)]
    [False Negatives (FN)  True Positives (TP) ]
"""

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for headless scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Reconfigure stdout for Windows terminal compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Import data loading & preprocessing function
from preprocess_data import load_and_preprocess

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
EVAL_DIR = MODEL_DIR / "evaluation"


def train_and_evaluate():
    # Ensure directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("[*] STEP 5: MACHINE LEARNING MODEL TRAINING & EVALUATION")
    print("=" * 70)
    print("Note: Educational ML demonstration - Not for clinical diagnosis.\n")

    # 1. Load preprocessed, clean, stratified data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, _ = load_and_preprocess(export_files=False)

    n_train = len(X_train)
    n_test = len(X_test)
    print(f"Dataset Split Overview:")
    print(f"  - Training samples: {n_train} (80%)")
    print(f"  - Testing samples:  {n_test} (20%)")
    print(f"  - Number of input features: {X_train.shape[1]}\n")

    # 2. Define the three models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42),
            "X_tr": X_train_scaled,
            "X_te": X_test_scaled,
            "requires_scaling": True,
            "description": "Linear probabilistic classifier trained on StandardScaler-normalized features."
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "X_tr": X_train,
            "X_te": X_test,
            "requires_scaling": False,
            "description": "Rule-based decision tree using raw unscaled features."
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "X_tr": X_train,
            "X_te": X_test,
            "requires_scaling": False,
            "description": "Ensemble of 100 decision trees voting on raw unscaled features."
        }
    }

    results = []
    confusion_matrices = {}

    # 3. Train and evaluate each model
    for name, config in models.items():
        clf = config["model"]
        X_tr = config["X_tr"]
        X_te = config["X_te"]

        # Train model on training data only
        clf.fit(X_tr, y_train)

        # Generate predictions on unseen test set
        y_pred = clf.predict(X_te)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Healthy (0)", "Disease (1)"])

        confusion_matrices[name] = cm

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Confusion Matrix": cm,
            "Classification Report": report,
            "Config": config
        })

    # 4. Print detailed reports for each model
    for res in results:
        name = res["Model"]
        cm = res["Confusion Matrix"]
        print("-" * 70)
        print(f"MODEL: {name}")
        print("-" * 70)
        print(f"Description: {res['Config']['description']}")
        print(f"Accuracy:  {res['Accuracy']:.4f} ({res['Accuracy']*100:.2f}%)")
        print(f"Precision: {res['Precision']:.4f} ({res['Precision']*100:.2f}%)")
        print(f"Recall:    {res['Recall']:.4f} ({res['Recall']*100:.2f}%)")
        print(f"F1 Score:  {res['F1 Score']:.4f} ({res['F1 Score']*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(f"  True Negatives (Healthy correct):   {cm[0, 0]}")
        print(f"  False Positives (Healthy as Sick):  {cm[0, 1]}")
        print(f"  False Negatives (Sick as Healthy):  {cm[1, 0]}")
        print(f"  True Positives (Sick correct):      {cm[1, 1]}")
        print("\nFull Classification Report:")
        print(res["Classification Report"])

    # 5. Model Comparison Table
    print("=" * 70)
    print("MODEL COMPARISON TABLE (Measured on 61-sample test set):")
    print("=" * 70)
    summary_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy']:.4f}",
            "Precision": f"{r['Precision']:.4f}",
            "Recall": f"{r['Recall']:.4f}",
            "F1 Score": f"{r['F1 Score']:.4f}"
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))

    # 6. Select model with highest test F1-score for this educational project
    best_result = max(results, key=lambda x: x["F1 Score"])
    selected_name = best_result["Model"]
    selected_clf = best_result["Config"]["model"]
    requires_scaling = best_result["Config"]["requires_scaling"]

    print("\n" + "=" * 70)
    print("SELECTED MODEL FOR EDUCATIONAL PROJECT:")
    print("=" * 70)
    print(f"Selected: {selected_name}")
    print(f"Reason:   Highest Test F1-Score ({best_result['F1 Score']:.4f})")
    print(f"Metrics:  Accuracy={best_result['Accuracy']:.4f}, Precision={best_result['Precision']:.4f}, Recall={best_result['Recall']:.4f}, F1={best_result['F1 Score']:.4f}")

    # 7. Save model and preprocessing artifacts
    saved_model_path = MODEL_DIR / "heart_disease_model.joblib"
    joblib.dump(selected_clf, saved_model_path)
    print(f"\n[+] Saved Trained Model to: {saved_model_path}")

    # Re-save scaler to ensure it is in sync
    saved_scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, saved_scaler_path)
    print(f"[+] Saved Scaler to:        {saved_scaler_path}")

    # Save model metadata JSON for Flask backend integration
    metadata = {
        "model_name": selected_name,
        "requires_scaling": requires_scaling,
        "features": list(X_train.columns),
        "target": "target",
        "training_samples": n_train,
        "test_samples": n_test,
        "metrics": {
            "accuracy": round(float(best_result["Accuracy"]), 4),
            "precision": round(float(best_result["Precision"]), 4),
            "recall": round(float(best_result["Recall"]), 4),
            "f1_score": round(float(best_result["F1 Score"]), 4)
        },
        "disclaimer": "This model is an educational machine-learning demonstration and is NOT intended for medical diagnosis."
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved Model Metadata to: {MODEL_DIR / 'model_metadata.json'}")

    # 8. Visualizations: Confusion Matrices
    print("\nGenerating confusion matrix visualizations in model/evaluation/ ...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Healthy (0)", "Disease (1)"],
            yticklabels=["Healthy (0)", "Disease (1)"],
            ax=ax
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Save individual confusion matrix plot as well
        indiv_fig, indiv_ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Healthy (0)", "Disease (1)"],
            yticklabels=["Healthy (0)", "Disease (1)"],
            ax=indiv_ax
        )
        indiv_ax.set_title(f"{name} - Confusion Matrix", fontweight="bold")
        indiv_ax.set_xlabel("Predicted Label")
        indiv_ax.set_ylabel("True Label")
        clean_filename = name.lower().replace(" ", "_")
        indiv_path = EVAL_DIR / f"confusion_matrix_{clean_filename}.png"
        indiv_fig.tight_layout()
        indiv_fig.savefig(indiv_path, dpi=150)
        plt.close(indiv_fig)

    all_path = EVAL_DIR / "confusion_matrix_all_models.png"
    fig.tight_layout()
    fig.savefig(all_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved evaluation plots to: {EVAL_DIR}")

    print("\n" + "=" * 70)
    print("[*] Model training, evaluation, and export completed successfully.")
    print("=" * 70)

    return results, best_result


if __name__ == "__main__":
    train_and_evaluate()
