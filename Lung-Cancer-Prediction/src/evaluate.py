"""
Evaluation Module for Lung Cancer Classification.
Computes comprehensive clinical validation metrics: Sensitivity (Recall),
Specificity, ROC-AUC, F1-score, Confusion Matrix, and generates publication-ready plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
)
from typing import Dict, Any, Optional


def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes diagnostic and statistical performance metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted positive class probabilities (optional).
        
    Returns:
        Dict[str, Any]: Dictionary containing all clinical metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    roc_auc = None
    if y_prob is not None:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        
    return {
        "accuracy": round(accuracy, 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "mcc": round(mcc, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn)
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str, model_name: str = "Classifier") -> None:
    """Generates and saves styled Confusion Matrix plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No Cancer (0)", "Cancer (1)"],
        yticklabels=["No Cancer (0)", "Cancer (1)"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Predicted Clinical Label", fontsize=11, labelpad=8)
    plt.ylabel("True Clinical Label", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, model_name: str = "Classifier") -> None:
    """Generates and saves ROC-AUC curve plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"{model_name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title(f"ROC-AUC Curve - {model_name}", fontsize=13, weight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(feature_names: list, importances: np.ndarray, save_path: str, top_n: int = 12) -> None:
    """Plots top predictive feature importances."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_imp, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Predictive Features in Lung Cancer Diagnosis", fontsize=13, weight="bold")
    plt.xlabel("Relative Importance Weight", fontsize=11)
    plt.ylabel("Clinical / Engineered Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
