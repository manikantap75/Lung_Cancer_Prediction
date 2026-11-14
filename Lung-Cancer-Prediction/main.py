"""
Main Execution Script for Lung Cancer Prediction Project.
Provides a unified entry point to run training, evaluation, unit tests,
sample inference, or launch the interactive Streamlit clinical web app.
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))


def run_training():
    """Runs data loading, clinical feature engineering, model training, and evaluation."""
    from train import run_training_pipeline
    data_file = os.path.join(PROJECT_ROOT, "data", "raw", "survey_lung_cancer.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    print("\n>>> Launching End-to-End Machine Learning Pipeline...")
    run_training_pipeline(data_file, models_dir, reports_dir)


def run_tests():
    """Executes automated unit test suite."""
    print("\n>>> Running Automated Unit Tests...")
    test_file = os.path.join(PROJECT_ROOT, "tests", "test_pipeline.py")
    result = subprocess.run([sys.executable, "-m", "unittest", test_file])
    sys.exit(result.returncode)


def run_prediction_demo():
    """Runs sample diagnostic predictions for high and low risk clinical profiles."""
    from predict import LungCancerPredictor
    print("\n>>> Initializing Inference Service...")
    predictor = LungCancerPredictor()

    patient_case = {
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
    result = predictor.predict_single(patient_case)
    print("\n" + "=" * 55)
    print("      SAMPLE CLINICAL DIAGNOSTIC PREDICTION")
    print("=" * 55)
    print(f"  Classification : {result['prediction_label']}")
    print(f"  Malignancy Prob: {result['risk_probability'] * 100:.2f}%")
    print(f"  Risk Category  : {result['risk_tier']}")
    print("  Clinical Guidance:")
    for g in result["clinical_guidance"]:
        print(f"    - {g}")
    print("=" * 55 + "\n")


def launch_app():
    """Launches the Streamlit clinical decision support dashboard."""
    app_file = os.path.join(PROJECT_ROOT, "app", "app.py")
    print(f"\n>>> Starting Streamlit Web App at {app_file}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Cancer Prediction CLI")
    parser.add_argument(
        "--action",
        choices=["train", "test", "predict", "app"],
        default="predict",
        help="Action to perform: 'train' (re-train models), 'test' (run unit tests), 'predict' (demo inference), 'app' (launch Streamlit UI)"
    )
    args = parser.parse_args()

    if args.action == "train":
        run_training()
    elif args.action == "test":
        run_tests()
    elif args.action == "predict":
        run_prediction_demo()
    elif args.action == "app":
        launch_app()
