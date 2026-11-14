"""
Automated Unit Tests for Lung Cancer ML Pipeline.
Validates data loading, preprocessing transformations, feature engineering, and inference engine.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure src directory is accessible
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data_loader import load_raw_data, get_data_summary
from preprocessor import ClinicalDataPreprocessor, encode_target
from feature_engineering import ClinicalFeatureEngineer
from predict import LungCancerPredictor


class TestLungCancerPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(PROJECT_ROOT, "data", "raw", "survey_lung_cancer.csv")
        cls.df_raw = load_raw_data(cls.data_path)
        cls.predictor = LungCancerPredictor()

    def test_01_data_loading(self):
        """Verify dataset loads properly and has required clinical columns."""
        self.assertGreater(len(self.df_raw), 50)
        required_cols = ["GENDER", "AGE", "SMOKING", "COUGHING", "CHEST_PAIN", "LUNG_CANCER"]
        for col in required_cols:
            self.assertIn(col, self.df_raw.columns)

    def test_02_data_summary(self):
        """Verify summary diagnostic dictionary."""
        summary = get_data_summary(self.df_raw)
        self.assertEqual(summary["missing_values"], 0)
        self.assertIn("target_distribution", summary)

    def test_03_preprocessor_encoding(self):
        """Verify binary encoding and gender mapping."""
        sample_df = pd.DataFrame({
            "GENDER": ["M", "F"],
            "AGE": [55, 62],
            "SMOKING": [2, 1],
            "COUGHING": [2, 1]
        })
        prep = ClinicalDataPreprocessor(scale_features=False)
        prep.fit(sample_df)
        transformed = prep.transform(sample_df)
        
        # Gender should be mapped M->1, F->0
        self.assertEqual(list(transformed["GENDER"]), [1, 0])
        # 1/2 survey scale should be mapped to 0/1 binary
        self.assertEqual(list(transformed["SMOKING"]), [1, 0])
        self.assertEqual(list(transformed["COUGHING"]), [1, 0])

    def test_04_target_encoding(self):
        """Verify target encoding handles YES/NO cleanly."""
        targets = pd.Series(["YES", "NO", "YES"])
        encoded = encode_target(targets)
        self.assertEqual(list(encoded), [1, 0, 1])

    def test_05_feature_engineering(self):
        """Verify domain engineered features are computed correctly."""
        engineer = ClinicalFeatureEngineer()
        sample_df = pd.DataFrame({
            "GENDER": ["M"],
            "AGE": [60],
            "SMOKING": [2],
            "YELLOW_FINGERS": [2],
            "ALCOHOL_CONSUMING": [2],
            "WHEEZING": [2],
            "COUGHING": [2],
            "SHORTNESS_OF_BREATH": [2],
            "FATIGUE": [2],
            "SWALLOWING_DIFFICULTY": [2],
            "CHEST_PAIN": [2],
            "CHRONIC_DISEASE": [2]
        })
        df_eng = engineer.transform(sample_df)
        
        # Check all 5 engineered features exist
        self.assertIn("RESPIRATORY_SYMPTOM_SCORE", df_eng.columns)
        self.assertIn("LIFESTYLE_RISK_SCORE", df_eng.columns)
        self.assertIn("THORACIC_COMPLICATION_SCORE", df_eng.columns)
        self.assertIn("TOTAL_SYMPTOM_BURDEN", df_eng.columns)
        self.assertIn("AGE_SMOKING_RISK", df_eng.columns)

        # In this all-positive symptom case:
        # RESPIRATORY_SYMPTOM_SCORE = WHEEZING (1) + COUGHING (1) + SHORTNESS_OF_BREATH (1) = 3
        self.assertEqual(df_eng["RESPIRATORY_SYMPTOM_SCORE"].iloc[0], 3)

    def test_06_single_inference(self):
        """Verify inference engine outputs probability, label, and risk tier."""
        test_case = {
            "GENDER": "M",
            "AGE": 65,
            "SMOKING": 2,
            "YELLOW_FINGERS": 2,
            "ANXIETY": 1,
            "PEER_PRESSURE": 2,
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
        res = self.predictor.predict_single(test_case)
        self.assertIn("risk_probability", res)
        self.assertGreaterEqual(res["risk_probability"], 0.0)
        self.assertLessEqual(res["risk_probability"], 1.0)
        self.assertIn(res["prediction"], [0, 1])
        self.assertIn("clinical_guidance", res)
        self.assertGreater(len(res["clinical_guidance"]), 0)

    def test_07_batch_inference(self):
        """Verify batch inference appends risk tier and probabilities."""
        batch_sample = self.df_raw.head(5).drop(columns=["LUNG_CANCER"])
        scored_df = self.predictor.predict_batch(batch_sample)
        self.assertEqual(len(scored_df), 5)
        self.assertIn("RISK_PROBABILITY", scored_df.columns)
        self.assertIn("RISK_TIER", scored_df.columns)


if __name__ == "__main__":
    unittest.main()
