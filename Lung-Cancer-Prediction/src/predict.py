"""
Inference & Prediction Service for Lung Cancer Early Diagnosis.
Loads serialized pipeline artifacts and evaluates clinical patient risk profiles.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List

from preprocessor import load_preprocessor, ClinicalDataPreprocessor
from feature_engineering import ClinicalFeatureEngineer


class LungCancerPredictor:
    """
    Production inference engine for Lung Cancer risk classification.
    """
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(project_root, "models", "best_lung_cancer_model.pkl")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")
            
        self.model = joblib.load(model_path)
        self.preprocessor = load_preprocessor(preprocessor_path)
        self.feature_engineer = ClinicalFeatureEngineer()
        
    def _assess_risk_tier(self, probability: float) -> str:
        """Categorizes raw probability into actionable clinical risk tiers."""
        if probability < 0.35:
            return "Low Risk"
        elif probability <= 0.70:
            return "Moderate Risk"
        else:
            return "High Risk (Clinical Follow-up Recommended)"

    def _generate_clinical_guidance(self, patient_dict: Dict[str, Any], probability: float) -> List[str]:
        """Provides tailored medical and screening suggestions based on patient presentation."""
        guidance = []
        if probability >= 0.70:
            guidance.append("Schedule Low-Dose Computed Tomography (LDCT) thoracic screening.")
            guidance.append("Urgent consultation with a Pulmonologist / Thoracic Oncology specialist.")
        elif probability >= 0.35:
            guidance.append("Secondary diagnostic evaluation and symptom monitoring recommended in 3-6 months.")
            guidance.append("Spirometry or chest radiography to evaluate airway symptoms.")
        else:
            guidance.append("Routine annual health check-up. Continue maintaining healthy lifestyle habits.")
            
        # Lifestyle-specific notes
        smoking_val = str(patient_dict.get("SMOKING", "")).upper()
        if smoking_val in ["2", "YES", "TRUE"]:
            guidance.append("Referral to structured tobacco cessation program.")
            
        alcohol_val = str(patient_dict.get("ALCOHOL_CONSUMING", "")).upper()
        if alcohol_val in ["2", "YES", "TRUE"]:
            guidance.append("Alcohol moderation counseling.")
            
        return guidance

    def predict_single(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs diagnostic prediction for an individual patient profile.
        
        Args:
            patient_data: Dict of clinical attributes (GENDER, AGE, SMOKING, etc.)
            
        Returns:
            Dict[str, Any]: Risk probability, predicted classification, tier, and recommendations.
        """
        df_single = pd.DataFrame([patient_data])
        df_single.columns = df_single.columns.str.strip().str.upper()
        
        # 1. Feature Engineering
        df_eng = self.feature_engineer.transform(df_single)
        
        # 2. Preprocessing & Scaling
        df_processed = self.preprocessor.transform(df_eng)
        
        # 3. Model Inference
        prob = float(self.model.predict_proba(df_processed)[0][1])
        prediction = int(self.model.predict(df_processed)[0])
        risk_tier = self._assess_risk_tier(prob)
        recommendations = self._generate_clinical_guidance(patient_data, prob)
        
        # Identify prominent symptoms reported
        reported_symptoms = [
            k for k, v in patient_data.items()
            if str(v).upper() in ["2", "YES", "TRUE"] and k not in ["GENDER", "AGE", "LUNG_CANCER"]
        ]
        
        return {
            "prediction": prediction,
            "prediction_label": "Positive (High Risk)" if prediction == 1 else "Negative (Low Risk)",
            "risk_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "reported_symptoms": reported_symptoms,
            "clinical_guidance": recommendations
        }

    def predict_batch(self, df_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Runs bulk diagnostic predictions on a batch patient dataset.
        """
        df = df_patients.copy()
        df.columns = df.columns.str.strip().str.upper()
        
        # Feature Engineering
        df_eng = self.feature_engineer.transform(df)
        
        # Preprocessing
        df_processed = self.preprocessor.transform(df_eng)
        
        probs = self.model.predict_proba(df_processed)[:, 1]
        preds = self.model.predict(df_processed)
        
        result_df = df_patients.copy()
        result_df["PREDICTED_CLASS"] = preds
        result_df["RISK_PROBABILITY"] = np.round(probs, 4)
        result_df["RISK_TIER"] = [self._assess_risk_tier(p) for p in probs]
        return result_df


if __name__ == "__main__":
    predictor = LungCancerPredictor()
    
    # Test High-risk synthetic case
    high_risk_patient = {
        "GENDER": "M",
        "AGE": 68,
        "SMOKING": 2,
        "YELLOW_FINGERS": 2,
        "ANXIETY": 1,
        "PEER_PRESSURE": 1,
        "CHRONIC_DISEASE": 2,
        "FATIGUE": 2,
        "ALLERGY": 1,
        "WHEEZING": 2,
        "ALCOHOL_CONSUMING": 2,
        "COUGHING": 2,
        "SHORTNESS_OF_BREATH": 2,
        "SWALLOWING_DIFFICULTY": 2,
        "CHEST_PAIN": 2
    }
    
    result = predictor.predict_single(high_risk_patient)
    print("\n--- Diagnostic Prediction Result (High Risk Case) ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
        
    # Test Low-risk synthetic case
    low_risk_patient = {
        "GENDER": "F",
        "AGE": 32,
        "SMOKING": 1,
        "YELLOW_FINGERS": 1,
        "ANXIETY": 1,
        "PEER_PRESSURE": 1,
        "CHRONIC_DISEASE": 1,
        "FATIGUE": 1,
        "ALLERGY": 1,
        "WHEEZING": 1,
        "ALCOHOL_CONSUMING": 1,
        "COUGHING": 1,
        "SHORTNESS_OF_BREATH": 1,
        "SWALLOWING_DIFFICULTY": 1,
        "CHEST_PAIN": 1
    }
    result_low = predictor.predict_single(low_risk_patient)
    print("\n--- Diagnostic Prediction Result (Low Risk Case) ---")
    for k, v in result_low.items():
        print(f"  {k}: {v}")
