"""
Preprocessing Module for Clinical Lung Cancer Dataset.
Standardizes categorical variables, encodes binary symptoms, and scales numeric features.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


SYMPTOM_COLUMNS = [
    "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING",
    "ALCOHOL_CONSUMING", "COUGHING", "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
]


class ClinicalDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to normalize clinical inputs for Lung Cancer Modeling.
    Converts 1/2 clinical survey scale to 0/1 binary scale, encodes Gender,
    and applies standard scaling to Age and composite scores.
    """
    def __init__(self, scale_features: bool = True):
        self.scale_features = scale_features
        self.scaler = StandardScaler()
        self.numeric_cols = ["AGE"]
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # Find which numeric columns are present to fit scaler
        fit_cols = [c for c in self.numeric_cols if c in X.columns]
        if self.scale_features and fit_cols:
            self.scaler.fit(X[fit_cols])
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        
        X = X.copy()
        
        # 1. Normalize Gender (M -> 1, F -> 0)
        if "GENDER" in X.columns:
            X["GENDER"] = X["GENDER"].astype(str).str.upper().map({"M": 1, "F": 0}).fillna(0).astype(int)
            
        # 2. Normalize 1/2 survey symptoms into 0/1 binary indicators
        for col in SYMPTOM_COLUMNS:
            if col in X.columns:
                # If values are 1 and 2: map 1 -> 0 (Absent), 2 -> 1 (Present)
                # If values are already 0 and 1: keep as is
                if set(X[col].dropna().unique()).issubset({1, 2}):
                    X[col] = X[col].map({1: 0, 2: 1})
                X[col] = X[col].fillna(0).astype(int)
                
        # 3. Scale numeric features if requested
        fit_cols = [c for c in self.numeric_cols if c in X.columns]
        if self.scale_features and fit_cols:
            X[fit_cols] = self.scaler.transform(X[fit_cols])
            
        return X


def encode_target(series: pd.Series) -> pd.Series:
    """
    Encodes LUNG_CANCER outcome into binary integer (YES=1, NO=0).
    """
    cleaned = series.astype(str).str.strip().str.upper()
    return cleaned.map({"YES": 1, "NO": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def save_preprocessor(preprocessor: ClinicalDataPreprocessor, filepath: str) -> None:
    """Serializes fitted preprocessor to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)


def load_preprocessor(filepath: str) -> ClinicalDataPreprocessor:
    """Loads serialized preprocessor from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor artifact not found at {filepath}")
    return joblib.load(filepath)
