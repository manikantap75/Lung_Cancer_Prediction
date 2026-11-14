"""
Feature Engineering Module for Clinical Lung Cancer Prediction.
Creates domain-specific oncology risk indicators, symptom burden indices,
and risk interaction terms to boost classifier discriminative power.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs clinical risk indices
    from raw or preprocessed patient survey responses.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates domain-engineered features on patient data.
        Assumes binary symptom columns are 0/1 or 1/2.
        """
        df = X.copy()

        # Helper to ensure 0/1 representation for calculations
        def get_binary_val(series):
            if set(series.dropna().unique()).issubset({1, 2}):
                return series.map({1: 0, 2: 1}).fillna(0)
            return series.fillna(0)

        smoking_bin = get_binary_val(df["SMOKING"]) if "SMOKING" in df.columns else 0
        yellow_fingers_bin = get_binary_val(df["YELLOW_FINGERS"]) if "YELLOW_FINGERS" in df.columns else 0
        alcohol_bin = get_binary_val(df["ALCOHOL_CONSUMING"]) if "ALCOHOL_CONSUMING" in df.columns else 0
        wheezing_bin = get_binary_val(df["WHEEZING"]) if "WHEEZING" in df.columns else 0
        coughing_bin = get_binary_val(df["COUGHING"]) if "COUGHING" in df.columns else 0
        sob_bin = get_binary_val(df["SHORTNESS_OF_BREATH"]) if "SHORTNESS_OF_BREATH" in df.columns else 0
        fatigue_bin = get_binary_val(df["FATIGUE"]) if "FATIGUE" in df.columns else 0
        swallow_bin = get_binary_val(df["SWALLOWING_DIFFICULTY"]) if "SWALLOWING_DIFFICULTY" in df.columns else 0
        chest_pain_bin = get_binary_val(df["CHEST_PAIN"]) if "CHEST_PAIN" in df.columns else 0
        chronic_bin = get_binary_val(df["CHRONIC_DISEASE"]) if "CHRONIC_DISEASE" in df.columns else 0

        # 1. Pulmonary / Respiratory Symptom Score (0 - 3)
        df["RESPIRATORY_SYMPTOM_SCORE"] = wheezing_bin + coughing_bin + sob_bin

        # 2. Lifestyle Carcinogen Risk Index (Weighted exposure)
        df["LIFESTYLE_RISK_SCORE"] = (smoking_bin * 2) + alcohol_bin + yellow_fingers_bin

        # 3. Thoracic & Mediastinal Complication Score (Dysphagia + Chest Pain + Fatigue)
        df["THORACIC_COMPLICATION_SCORE"] = swallow_bin + chest_pain_bin + fatigue_bin

        # 4. Total Symptom Count (Aggregated burden across all assessed symptoms)
        symptom_cols = [
            smoking_bin, yellow_fingers_bin, alcohol_bin, wheezing_bin,
            coughing_bin, sob_bin, fatigue_bin, swallow_bin, chest_pain_bin, chronic_bin
        ]
        df["TOTAL_SYMPTOM_BURDEN"] = sum(symptom_cols)

        # 5. Age-Smoking Cumulative Interaction (if AGE column exists)
        if "AGE" in df.columns:
            # Normalize age relative to mean clinical threshold (60)
            df["AGE_SMOKING_RISK"] = (df["AGE"] / 60.0) * (smoking_bin + 0.5)
        else:
            df["AGE_SMOKING_RISK"] = smoking_bin

        return df


def generate_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience functional wrapper for feature engineering."""
    engineer = ClinicalFeatureEngineer()
    return engineer.transform(df)


if __name__ == "__main__":
    from data_loader import load_raw_data
    import os
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "survey_lung_cancer.csv")
    df = load_raw_data(raw_path)
    df_eng = generate_engineered_features(df)
    print("Engineered Features Sample:")
    eng_cols = ["RESPIRATORY_SYMPTOM_SCORE", "LIFESTYLE_RISK_SCORE", "THORACIC_COMPLICATION_SCORE", "TOTAL_SYMPTOM_BURDEN", "AGE_SMOKING_RISK"]
    print(df_eng[eng_cols].head())
