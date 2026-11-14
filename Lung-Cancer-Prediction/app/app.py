"""
Streamlit Web Application: Lung Cancer Early Diagnosis & Risk Stratification Dashboard.
Provides an interactive clinical interface for individual patient screening, batch analysis,
and model interpretability visualizations.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Ensure src directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "src"))

from predict import LungCancerPredictor


# Page Configuration
st.set_page_config(
    page_title="Lung Cancer Early Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Healthcare UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .risk-mod {
        background-color: #FEF3C7;
        border-left: 6px solid #D97706;
        padding: 1.2rem;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 6px solid #059669;
        padding: 1.2rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Initializes and caches the trained prediction pipeline."""
    return LungCancerPredictor()


@st.cache_data
def get_metrics_summary():
    """Loads saved validation metrics and feature importances."""
    metrics_path = os.path.join(project_root, "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    predictor = get_predictor()
    metrics = get_metrics_summary()

    # Header
    st.markdown('<div class="main-header">🫁 Clinical Lung Cancer Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">An AI-assisted diagnostic decision support system leveraging supervised machine learning & clinical feature engineering.</div>',
        unsafe_allow_html=True
    )

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔬 Individual Patient Assessment",
        "📂 Batch Clinical Screening",
        "📊 Model Performance & Explainability"
    ])

    # =========================================================================
    # TAB 1: INDIVIDUAL PATIENT ASSESSMENT
    # =========================================================================
    with tab1:
        st.subheader("Patient Clinical & Demographic Profile")
        st.write("Input patient clinical survey responses to compute oncology risk stratification.")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("#### 👤 Demographics & Lifestyle")
            gender = st.selectbox("Biological Sex", ["Male", "Female"], help="Gender assigned at birth.")
            age = st.slider("Age (Years)", min_value=18, max_value=95, value=58, step=1)
            smoking = st.selectbox("Active Tobacco Smoking?", ["No", "Yes"], index=1)
            yellow_fingers = st.selectbox("Yellow Finger Staining (Nicotine)?", ["No", "Yes"], index=1)
            alcohol = st.selectbox("Frequent Alcohol Consumption?", ["No", "Yes"], index=0)

        with col2:
            st.markdown("#### 🫁 Respiratory Symptoms")
            coughing = st.selectbox("Persistent Coughing?", ["No", "Yes"], index=1)
            wheezing = st.selectbox("Audible Wheezing / Stridor?", ["No", "Yes"], index=1)
            sob = st.selectbox("Shortness of Breath (Dyspnea)?", ["No", "Yes"], index=1)
            chest_pain = st.selectbox("Chest Pain / Thoracic Discomfort?", ["No", "Yes"], index=1)
            swallowing = st.selectbox("Swallowing Difficulty (Dysphagia)?", ["No", "Yes"], index=0)

        with col3:
            st.markdown("#### 🩺 Systemic & Psychological")
            fatigue = st.selectbox("Chronic Fatigue / Malaise?", ["No", "Yes"], index=1)
            chronic_disease = st.selectbox("Chronic Pulmonary / Systemic Disease?", ["No", "Yes"], index=0)
            allergy = st.selectbox("Known Respiratory Allergies?", ["No", "Yes"], index=0)
            anxiety = st.selectbox("Anxiety Disorders?", ["No", "Yes"], index=0)
            peer_pressure = st.selectbox("Environmental / Peer Smoking Exposure?", ["No", "Yes"], index=0)

        # Mapping UI values to model format (1 = No, 2 = Yes; Gender: M/F)
        val_map = {"No": 1, "Yes": 2}
        patient_payload = {
            "GENDER": "M" if gender == "Male" else "F",
            "AGE": age,
            "SMOKING": val_map[smoking],
            "YELLOW_FINGERS": val_map[yellow_fingers],
            "ANXIETY": val_map[anxiety],
            "PEER_PRESSURE": val_map[peer_pressure],
            "CHRONIC_DISEASE": val_map[chronic_disease],
            "FATIGUE": val_map[fatigue],
            "ALLERGY": val_map[allergy],
            "WHEEZING": val_map[wheezing],
            "ALCOHOL_CONSUMING": val_map[alcohol],
            "COUGHING": val_map[coughing],
            "SHORTNESS_OF_BREATH": val_map[sob],
            "SWALLOWING_DIFFICULTY": val_map[swallowing],
            "CHEST_PAIN": val_map[chest_pain]
        }

        st.markdown("---")
        assess_button = st.button("🚀 Run Risk Stratification Analysis", type="primary", use_container_width=True)

        if assess_button:
            result = predictor.predict_single(patient_payload)
            prob = result["risk_probability"]
            prob_percent = prob * 100.0
            tier = result["risk_tier"]

            st.markdown("### Diagnostic Assessment Summary")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prob >= 0.70:
                    st.markdown(f"""
                    <div class="risk-high">
                        <h3 style="color: #991B1B; margin-top:0;">⚠️ {tier}</h3>
                        <p style="font-size: 1.1rem; color: #7F1D1D;">
                            Calculated Malignancy Probability: <strong>{prob_percent:.1f}%</strong>
                        </p>
                        <p style="color: #991B1B;">
                            High probability of pulmonary neoplasm based on cumulative symptom presentation and risk factors.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown(f"""
                    <div class="risk-mod">
                        <h3 style="color: #92400E; margin-top:0;">⚡ {tier}</h3>
                        <p style="font-size: 1.1rem; color: #78350F;">
                            Calculated Malignancy Probability: <strong>{prob_percent:.1f}%</strong>
                        </p>
                        <p style="color: #92400E;">
                            Borderline clinical risk markers observed. Closer monitoring or secondary testing suggested.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <h3 style="color: #065F46; margin-top:0;">✅ {tier}</h3>
                        <p style="font-size: 1.1rem; color: #047857;">
                            Calculated Malignancy Probability: <strong>{prob_percent:.1f}%</strong>
                        </p>
                        <p style="color: #065F46;">
                            Patient profile demonstrates low concordance with primary clinical indicators of lung cancer.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(prob)

            with res_col2:
                st.markdown("#### 📋 Recommended Clinical Actions")
                for rec in result["clinical_guidance"]:
                    st.markdown(f"- 🩺 {rec}")

                if result["reported_symptoms"]:
                    st.markdown("#### 🔍 Active Risk Flags Present")
                    st.write(", ".join([s.replace("_", " ").title() for s in result["reported_symptoms"]]))

            # Medical Report Generator
            report_text = f"""=====================================================
CLINICAL LUNG CANCER RISK ASSESSMENT REPORT
Generated via AI Machine Learning Diagnostic Decision Support
=====================================================
Patient Demographics:
  Sex: {gender} | Age: {age}
Diagnostic Assessment:
  Risk Classification: {result['prediction_label']}
  Predicted Malignancy Probability: {prob_percent:.2f}%
  Stratification Tier: {tier}

Active Risk Symptoms:
  {', '.join([s.replace('_', ' ').title() for s in result['reported_symptoms']]) if result['reported_symptoms'] else 'None reported'}

Clinical Recommendations:
""" + "\n".join([f"  - {r}" for r in result["clinical_guidance"]]) + """

Notice: This decision-support output is intended for healthcare professional screening
guidance and does not constitute a standalone histological diagnosis.
=====================================================
"""
            st.download_button(
                label="📥 Download Clinical Patient Report (.txt)",
                data=report_text,
                file_name=f"LungCancer_Risk_Report_Age{age}_{gender}.txt",
                mime="text/plain"
            )

    # =========================================================================
    # TAB 2: BATCH SCREENING
    # =========================================================================
    with tab2:
        st.subheader("Batch Patient Cohort Analysis")
        st.write("Upload a CSV file containing clinical records to perform automated mass risk screening.")

        sample_csv_path = os.path.join(project_root, "data", "raw", "survey_lung_cancer.csv")
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📄 Download Sample Clinical Template CSV",
                    data=f.read(),
                    file_name="lung_cancer_template.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df)}** patient records.")
            
            with st.spinner("Processing batch inference..."):
                scored_df = predictor.predict_batch(batch_df)
                
            st.success("Batch risk stratification complete!")
            st.dataframe(scored_df.head(20), use_container_width=True)

            # Summary Metrics for Batch
            high_count = (scored_df["RISK_TIER"].str.contains("High")).sum()
            mod_count = (scored_df["RISK_TIER"].str.contains("Moderate")).sum()
            low_count = (scored_df["RISK_TIER"].str.contains("Low")).sum()

            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("High Risk Patients", f"{high_count} ({high_count/len(scored_df)*100:.1f}%)")
            b_col2.metric("Moderate Risk Patients", f"{mod_count} ({mod_count/len(scored_df)*100:.1f}%)")
            b_col3.metric("Low Risk Patients", f"{low_count} ({low_count/len(scored_df)*100:.1f}%)")

            # Export scored dataset
            csv_export = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Cohort Dataset (CSV)",
                data=csv_export,
                file_name="lung_cancer_batch_predictions.csv",
                mime="text/csv"
            )

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & EXPLAINABILITY
    # =========================================================================
    with tab3:
        st.subheader("Model Validation & Clinical Explainability")
        st.write("Rigorous clinical validation metrics achieved via 5-Fold Stratified Cross-Validation on held-out test data.")

        if metrics:
            fm = metrics.get("final_metrics", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{fm.get('accuracy', 0)*100:.1f}%")
            m2.metric("Sensitivity (Recall)", f"{fm.get('sensitivity_recall', 0)*100:.1f}%", help="True Positive Rate (critically minimizes false negative oncology misses)")
            m3.metric("Specificity", f"{fm.get('specificity', 0)*100:.1f}%")
            m4.metric("Precision", f"{fm.get('precision', 0)*100:.1f}%")
            m5.metric("ROC-AUC", f"{fm.get('roc_auc', 0):.3f}")

        st.markdown("---")
        st.markdown("### Diagnostic Visualizations")
        fig_col1, fig_col2 = st.columns(2)

        reports_dir = os.path.join(project_root, "reports", "figures")
        cm_file = os.path.join(reports_dir, "confusion_matrix.png")
        roc_file = os.path.join(reports_dir, "roc_curve.png")
        feat_file = os.path.join(reports_dir, "feature_importance.png")
        comp_file = os.path.join(reports_dir, "model_comparison.png")

        with fig_col1:
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Confusion Matrix on Test Cohort", use_container_width=True)
            if os.path.exists(feat_file):
                st.image(feat_file, caption="Top Clinical Predictive Risk Factors", use_container_width=True)

        with fig_col2:
            if os.path.exists(roc_file):
                st.image(roc_file, caption="Receiver Operating Characteristic (ROC-AUC)", use_container_width=True)
            if os.path.exists(comp_file):
                st.image(comp_file, caption="Classifier Algorithm Benchmark Comparison", use_container_width=True)


if __name__ == "__main__":
    main()
