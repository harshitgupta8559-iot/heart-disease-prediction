"""
Dataset Inspection Script for Heart Disease Prediction
======================================================
This beginner-friendly script checks the 'dataset' folder for a CSV file,
loads it using pandas, and prints essential exploratory data analysis (EDA) details:
- File detection
- Row and column count (shape)
- Column names and data types
- Preview of the first 5 rows
- Missing / null value counts
- Duplicate row counts
- Target column detection and value counts
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure standard UTF-8 output if terminal allows or reconfigure stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

def inspect():
    print("=" * 60)
    print("[*] HEART DISEASE DATASET INSPECTION")
    print("=" * 60)

    # 1. Search for any CSV file inside the dataset/ folder
    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        print(f"\n[X] No CSV dataset found in: {DATASET_DIR}")
        print("Please place your heart disease CSV file inside the 'dataset' folder and run this script again.\n")
        return

    # Use the first CSV file found
    dataset_path = csv_files[0]
    print(f"\n[+] Found Dataset: {dataset_path.name}")
    print(f"[+] Full Path: {dataset_path}")

    # 2. Read the dataset using pandas
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"\n[X] Error reading CSV file: {e}")
        return

    # 3. Shape (Rows and Columns)
    rows, cols = df.shape
    print("\n" + "-" * 40)
    print("1. DATASET DIMENSIONS")
    print("-" * 40)
    print(f"Total Rows (Records):     {rows}")
    print(f"Total Columns (Features): {cols}")

    # 4. Column Names and Data Types
    print("\n" + "-" * 40)
    print("2. COLUMNS & DATA TYPES")
    print("-" * 40)
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })
    print(col_info.to_string(index=False))

    # 5. First 5 Rows Preview
    print("\n" + "-" * 40)
    print("3. FIRST 5 ROWS PREVIEW")
    print("-" * 40)
    print(df.head().to_string())

    # 6. Missing / Null Values Check
    print("\n" + "-" * 40)
    print("4. MISSING (NULL) VALUES CHECK")
    print("-" * 40)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("[+] No missing values found in the dataset! (0 nulls)")
    else:
        print(f"[!] Found {total_missing} missing value(s) in total:")
        print(missing[missing > 0])
    
    print("\nMissing values per column:")
    for col in df.columns:
        print(f"  - {col}: {df[col].isnull().sum()} missing")

    # 7. Duplicate Rows Check
    print("\n" + "-" * 40)
    print("5. DUPLICATE ROWS CHECK")
    print("-" * 40)
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        print("[+] No duplicate rows found.")
    else:
        print(f"[!] Found {duplicate_count} duplicate row(s).")

    # 8. Target / Output Column Identification
    print("\n" + "-" * 40)
    print("6. TARGET COLUMN INSPECTION")
    print("-" * 40)
    possible_targets = ["target", "output", "heartdisease", "condition", "num", "diagnosis", "HeartDisease"]
    target_col = None
    for col in df.columns:
        if col.lower() in [t.lower() for t in possible_targets]:
            target_col = col
            break

    if target_col:
        print(f"Identified Target Column: '{target_col}'")
        print("\nUnique Values and Counts:")
        counts = df[target_col].value_counts(dropna=False)
        for val, count in counts.items():
            pct = (count / len(df)) * 100
            print(f"  Value {val}: {count} rows ({pct:.2f}%)")
    else:
        print("[!] Could not automatically detect a target column name.")
        print(f"Available columns are: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("[*] Inspection Complete")
    print("=" * 60)

if __name__ == "__main__":
    inspect()
