"""
Data Loader Module for Lung Cancer Clinical Dataset.
Handles loading raw clinical survey records, schema normalization, and exploratory checks.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and standardizes column headers.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe with normalized column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    # Strip whitespace from column headers and capitalize
    df.columns = df.columns.str.strip().str.upper()
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary statistics and quality diagnostics for the clinical dataset.
    
    Args:
        df: Input clinical dataframe.
        
    Returns:
        Dict[str, Any]: Summary metadata including row count, class balance, missing values.
    """
    missing_count = int(df.isnull().sum().sum())
    duplicates_count = int(df.duplicated().sum())
    target_col = "LUNG_CANCER" if "LUNG_CANCER" in df.columns else None
    
    target_distribution = {}
    if target_col:
        target_distribution = df[target_col].value_counts(normalize=True).to_dict()
    
    return {
        "total_records": len(df),
        "total_features": len(df.columns),
        "missing_values": missing_count,
        "duplicate_rows": duplicates_count,
        "target_distribution": target_distribution,
        "columns": list(df.columns)
    }


if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "survey_lung_cancer.csv")
    df = load_raw_data(raw_path)
    summary = get_data_summary(df)
    print("Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
