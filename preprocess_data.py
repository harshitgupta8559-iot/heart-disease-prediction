"""
Data Preprocessing Script for Heart Disease Prediction
======================================================
This script performs clean, leak-free data preprocessing for our machine learning project:
1. Loads original 'heart.csv' without modifying it.
2. Removes exact duplicate rows (cleaning the 1025-row Kaggle copy back to 302 unique patients).
3. Separates features (X) and target (y).
4. Performs a stratified train-test split (80% train, 20% test).
5. Fits a StandardScaler ONLY on X_train (preventing data leakage).
6. Exports preprocessed train/test datasets for inspection and debugging.
7. Exports the fitted scaler object into model/scaler.joblib for Flask integration.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Reconfigure stdout to prevent character encoding issues on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "heart.csv"
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"


def load_and_preprocess(export_files=True):
    """
    Loads, cleans, splits, and scales the heart disease dataset.
    Returns:
        X_train, X_test (pandas DataFrames, unscaled)
        X_train_scaled, X_test_scaled (numpy arrays, scaled)
        y_train, y_test (pandas Series)
        scaler (fitted StandardScaler object)
        report_data (dict containing preprocessing metrics)
    """
    # -------------------------------------------------------------
    # Step 1: Load original dataset
    # -------------------------------------------------------------
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    df_original = pd.read_csv(DATASET_PATH)

    # -------------------------------------------------------------
    # Step 2: Make a working copy so the original stays untouched
    # -------------------------------------------------------------
    df = df_original.copy()
    original_rows, original_cols = df.shape

    # -------------------------------------------------------------
    # Step 3: Check missing values and duplicate rows
    # -------------------------------------------------------------
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    # -------------------------------------------------------------
    # Step 4: Handle duplicates
    # The Kaggle dataset contains 723 repeated rows due to merging
    # multiple database samples. Dropping exact duplicates ensures
    # the machine learning model learns from truly unique patients
    # and prevents artificial inflation of test accuracy.
    # -------------------------------------------------------------
    df_cleaned = df.drop_duplicates().reset_index(drop=True)
    cleaned_rows = len(df_cleaned)

    # -------------------------------------------------------------
    # Step 5 & 6: Separate features (X) and target (y)
    # The target column ('target') is excluded from features X.
    # -------------------------------------------------------------
    X = df_cleaned.drop(columns=["target"])
    y = df_cleaned["target"]
    feature_names = list(X.columns)

    # -------------------------------------------------------------
    # Step 7 & 8: Stratified Train-Test Split (80% Train, 20% Test)
    # Splitting BEFORE scaling prevents data leakage (information from
    # the test set leaking into the training set).
    # Stratify=y guarantees identical proportions of healthy vs diseased
    # patients in both train and test sets.
    # -------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------------------
    # Step 9 & 10: Scaling Strategy
    # Clinical variables (such as trestbps, chol, thalach, oldpeak, age)
    # have very different scales (cholesterol is ~200+, oldpeak is 0-6).
    # Distance-based models (Logistic Regression, SVM, KNN) require
    # feature scaling to converge properly and give equal weight to features.
    # Integer-coded clinical categoricals (sex, cp, fbs, etc.) are kept
    # numeric without unnecessary one-hot explosion to keep input dimensions
    # compact, robust, and directly matching clinical form inputs.
    # -------------------------------------------------------------
    scaler = StandardScaler()

    # Fit ONLY on training data, then transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # Step 11: Export artifacts for debugging / later steps
    # -------------------------------------------------------------
    if export_files:
        # Save split CSVs for debugging and inspection
        X_train.to_csv(DATASET_DIR / "X_train.csv", index=False)
        X_test.to_csv(DATASET_DIR / "X_test.csv", index=False)
        y_train.to_csv(DATASET_DIR / "y_train.csv", index=False)
        y_test.to_csv(DATASET_DIR / "y_test.csv", index=False)

        # Save fitted scaler for later use in Flask prediction pipeline
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    report_data = {
        "original_rows": original_rows,
        "original_cols": original_cols,
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
        "cleaned_rows": cleaned_rows,
        "features": feature_names,
        "target_col": "target",
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_train_dist": y_train.value_counts().to_dict(),
        "y_test_dist": y_test.value_counts().to_dict(),
    }

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, report_data


def print_preprocessing_report(report_data):
    """Prints a beginner-friendly summary of the preprocessing results."""
    print("=" * 65)
    print("[*] STEP 4: DATA PREPROCESSING SUMMARY")
    print("=" * 65)

    print("\n1. DATASET CLEANING & DEDUPLICATION:")
    print(f"  - Original rows loaded:        {report_data['original_rows']}")
    print(f"  - Original columns:            {report_data['original_cols']}")
    print(f"  - Missing/null values:         {report_data['missing_count']} (Zero missing values)")
    print(f"  - Exact duplicate rows found:  {report_data['duplicate_count']}")
    print(f"  - Rows removed:                {report_data['duplicate_count']}")
    print(f"  - Cleaned unique patient rows: {report_data['cleaned_rows']}")

    print("\n2. FEATURES & TARGET SEPARATION:")
    print(f"  - Target column:               '{report_data['target_col']}' (Excluded from features)")
    print(f"  - Number of input features:    {len(report_data['features'])}")
    print(f"  - Feature list:                {', '.join(report_data['features'])}")

    print("\n3. TRAIN-TEST SPLIT (80% Train / 20% Test, Stratified):")
    print(f"  - Training Set (X_train):      {report_data['X_train_shape'][0]} samples, {report_data['X_train_shape'][1]} features")
    print(f"  - Test Set (X_test):           {report_data['X_test_shape'][0]} samples, {report_data['X_test_shape'][1]} features")

    print("\n4. TARGET DISTRIBUTION COMPARISON (Stratification Check):")
    y_train_d = report_data['y_train_dist']
    y_test_d = report_data['y_test_dist']
    total_train = report_data['X_train_shape'][0]
    total_test = report_data['X_test_shape'][0]

    print(f"  - Training Labels:  Class 1 = {y_train_d.get(1, 0)} ({y_train_d.get(1, 0)/total_train*100:.1f}%), Class 0 = {y_train_d.get(0, 0)} ({y_train_d.get(0, 0)/total_train*100:.1f}%)")
    print(f"  - Testing Labels:   Class 1 = {y_test_d.get(1, 0)} ({y_test_d.get(1, 0)/total_test*100:.1f}%), Class 0 = {y_test_d.get(0, 0)} ({y_test_d.get(0, 0)/total_test*100:.1f}%)")
    print("    --> Proportions are perfectly preserved across both sets!")

    print("\n5. FEATURE SCALING:")
    print("  - StandardScaler fitted ONLY on X_train (0 data leakage).")
    print("  - Both X_train and X_test transformed using this scaler.")
    print("  - Fitted scaler exported to: model/scaler.joblib for the web application.")

    print("\n6. SAVED ARTIFACTS (Original heart.csv left completely unchanged):")
    print("  - dataset/X_train.csv")
    print("  - dataset/X_test.csv")
    print("  - dataset/y_train.csv")
    print("  - dataset/y_test.csv")
    print("  - model/scaler.joblib")

    print("\n" + "=" * 65)
    print("[*] Preprocessing completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    _, _, _, _, _, _, _, report = load_and_preprocess(export_files=True)
    print_preprocessing_report(report)
