"""
Model Training and Optimization Pipeline for Lung Cancer Prediction.
Performs cross-validation, hyperparameter tuning across multiple classifier algorithms,
selects the optimal estimator, and serializes production-ready artifacts.
"""

import os
import tempfile
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_raw_data
from preprocessor import ClinicalDataPreprocessor, encode_target, save_preprocessor
from feature_engineering import ClinicalFeatureEngineer
from evaluate import compute_clinical_metrics, plot_confusion_matrix, plot_roc_curve, plot_feature_importance


def run_training_pipeline(data_path: str, output_dir: str, reports_dir: str):
    """
    Executes the end-to-end model training, validation, and serialization workflow.
    """
    print("=" * 65)
    print("      LUNG CANCER PREDICTION - CLINICAL ML PIPELINE")
    print("=" * 65)
    
    # 1. Load Data
    print(f"\n[1/6] Loading raw clinical survey records from: {data_path}")
    raw_df = load_raw_data(data_path)
    print(f"      Total records loaded: {len(raw_df)}, Columns: {len(raw_df.columns)}")

    # 2. Extract Target and Features
    target_col = "LUNG_CANCER"
    y = encode_target(raw_df[target_col])
    X_raw = raw_df.drop(columns=[target_col])

    # 3. Apply Clinical Feature Engineering
    print("\n[2/6] Engineering domain-specific oncology risk indicators...")
    feature_engineer = ClinicalFeatureEngineer()
    X_engineered = feature_engineer.transform(X_raw)
    engineered_cols = [c for c in X_engineered.columns if c not in X_raw.columns]
    print(f"      New features generated: {engineered_cols}")

    # 4. Stratified Train-Test Split (80% Train, 20% Test)
    print("\n[3/6] Partitioning data into Stratified Train / Test sets (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train Set: {len(X_train_raw)} cases (Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)})")
    print(f"      Test Set:  {len(X_test_raw)} cases (Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)})")

    # 5. Fit Preprocessor on Train set only
    preprocessor = ClinicalDataPreprocessor(scale_features=True)
    # Include engineered numeric columns in scaling
    preprocessor.numeric_cols = ["AGE", "RESPIRATORY_SYMPTOM_SCORE", "LIFESTYLE_RISK_SCORE", 
                                 "THORACIC_COMPLICATION_SCORE", "TOTAL_SYMPTOM_BURDEN", "AGE_SMOKING_RISK"]
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 6. Evaluate Candidate Classifiers
    print("\n[4/6] Benchmarking multiple classifier algorithms (5-Fold Stratified CV)...")
    candidate_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Classifier (RBF)": SVC(probability=True, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_results = []

    for name, model in candidate_models.items():
        # Train on train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = compute_clinical_metrics(y_test.values, y_pred, y_prob)
        metrics["model_name"] = name
        comparison_results.append(metrics)
        print(f"      -> {name:<30}: Accuracy={metrics['accuracy']:.4f} | Recall={metrics['sensitivity_recall']:.4f} | ROC-AUC={metrics['roc_auc']} | F1={metrics['f1_score']:.4f}")

    # Plot Model Comparison Bar Chart
    comp_df = pd.DataFrame(comparison_results)
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ["accuracy", "sensitivity_recall", "f1_score"]
    melted = comp_df.melt(id_vars=["model_name"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="model_name", y="Score", hue="Metric", palette="Set2")
    plt.title("Comparative Performance Across Classifier Algorithms", fontsize=13, weight="bold")
    plt.xlabel("")
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0.5, 1.05)
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_plot_path = os.path.join(reports_dir, "model_comparison.png")
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()

    # 7. Hyperparameter Optimization on Top Classifier (Random Forest)
    print("\n[5/6] Performing GridSearch Hyperparameter Optimization on Random Forest...")
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    print(f"      Best Hyperparameters: {grid_search.best_params_}")
    print(f"      Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

    # 8. Final Evaluation of Tuned Best Model
    print("\n[6/6] Final Validation of Optimized Model on Holdout Test Set...")
    y_test_pred = best_estimator.predict(X_test)
    y_test_prob = best_estimator.predict_proba(X_test)[:, 1]
    final_metrics = compute_clinical_metrics(y_test.values, y_test_pred, y_test_prob)

    print("\n" + "=" * 65)
    print("                 FINAL VALIDATION METRICS")
    print("=" * 65)
    for k, v in final_metrics.items():
        print(f"  {k:<22}: {v}")
    print("=" * 65)

    # 9. Generate Diagnostic Visualizations
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_test.values, y_test_pred, cm_path, model_name="Random Forest (Optimized)")
    
    roc_path = os.path.join(reports_dir, "roc_curve.png")
    plot_roc_curve(y_test.values, y_test_prob, roc_path, model_name="Random Forest (Optimized)")

    feat_path = os.path.join(reports_dir, "feature_importance.png")
    feature_names = list(X_train.columns)
    importances = best_estimator.feature_importances_
    plot_feature_importance(feature_names, importances, feat_path, top_n=12)

    # 10. Persist Model, Preprocessor, and Metrics Artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_lung_cancer_model.pkl")
    joblib.dump(best_estimator, model_save_path)
    print(f"\n[Artifact Saved] Trained Model: {model_save_path}")

    prep_save_path = os.path.join(output_dir, "preprocessor.pkl")
    save_preprocessor(preprocessor, prep_save_path)
    print(f"[Artifact Saved] Preprocessor:  {prep_save_path}")

    metrics_save_path = os.path.join(output_dir, "metrics_summary.json")
    all_summary = {
        "final_metrics": final_metrics,
        "best_hyperparameters": grid_search.best_params_,
        "feature_names": feature_names,
        "feature_importances": dict(zip(feature_names, [round(float(x), 4) for x in importances])),
        "model_comparison": comparison_results
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=4)
    print(f"[Artifact Saved] Metrics Report: {metrics_save_path}")
    print("\nTraining Pipeline Completed Successfully!\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, "data", "raw", "survey_lung_cancer.csv")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports", "figures")

    run_training_pipeline(data_file, models_dir, reports_dir)
