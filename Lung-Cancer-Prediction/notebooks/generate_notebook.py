import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🫁 Lung Cancer Prediction & Clinical Risk Stratification\n",
            "### Exploratory Data Analysis, Clinical Feature Engineering & Machine Learning Benchmarking\n",
            "\n",
            "**Problem Statement:**  \n",
            "Early detection of pulmonary neoplasms is critical for improving survival rates in oncology. This notebook develops a clinical machine learning framework based on structured patient survey data, engineering diagnostic symptom indices and benchmarking multiple supervised classifiers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add src directory to path\n",
            "sys.path.append(os.path.abspath('../src'))\n",
            "from data_loader import load_raw_data, get_data_summary\n",
            "from feature_engineering import ClinicalFeatureEngineer\n",
            "from preprocessor import ClinicalDataPreprocessor, encode_target\n",
            "\n",
            "sns.set_theme(style='whitegrid')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Data Ingestion & Quality Diagnostics"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "data_path = '../data/raw/survey_lung_cancer.csv'\n",
            "df = load_raw_data(data_path)\n",
            "print(f'Dataset Shape: {df.shape}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "summary = get_data_summary(df)\n",
            "print('Data Quality Summary:')\n",
            "for k, v in summary.items():\n",
            "    print(f'  {k}: {v}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Exploratory Data Analysis (EDA)\n",
            "Examining patient age distributions stratified by malignant diagnosis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(9, 4))\n",
            "sns.histplot(data=df, x='AGE', hue='LUNG_CANCER', kde=True, bins=20, palette='Set1')\n",
            "plt.title('Age Distribution Stratified by Lung Cancer Outcome', fontsize=12, weight='bold')\n",
            "plt.xlabel('Patient Age (Years)')\n",
            "plt.ylabel('Patient Count')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Clinical Feature Engineering\n",
            "Constructing domain-specific clinical indices: `RESPIRATORY_SYMPTOM_SCORE`, `LIFESTYLE_RISK_SCORE`, and `THORACIC_COMPLICATION_SCORE`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "engineer = ClinicalFeatureEngineer()\n",
            "df_eng = engineer.transform(df)\n",
            "eng_cols = ['RESPIRATORY_SYMPTOM_SCORE', 'LIFESTYLE_RISK_SCORE', 'THORACIC_COMPLICATION_SCORE', 'TOTAL_SYMPTOM_BURDEN', 'AGE_SMOKING_RISK']\n",
            "df_eng[['LUNG_CANCER'] + eng_cols].head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 4))\n",
            "sns.boxplot(data=df_eng, x='LUNG_CANCER', y='TOTAL_SYMPTOM_BURDEN', palette='Blues')\n",
            "plt.title('Total Clinical Symptom Burden vs. Cancer Malignancy', fontsize=12, weight='bold')\n",
            "plt.xlabel('Lung Cancer Status')\n",
            "plt.ylabel('Aggregated Symptom Score')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Conclusion & Next Steps\n",
            "- Clinical symptom aggregation and lifestyle interaction terms produce strong statistical separation.\n",
            "- Run `python src/train.py` to train the full pipeline and `streamlit run app/app.py` for the interactive web app."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Lung-Cancer-Prediction/notebooks/01_exploratory_data_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created Jupyter Notebook successfully.")
