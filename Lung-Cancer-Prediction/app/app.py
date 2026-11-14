"""
Streamlit Web Application: Lung Cancer Early Diagnosis & Risk Stratification Dashboard.
Enhanced with a modern, high-aesthetic executive medical UI, glassmorphic card containers,
ambient gradient lighting, dynamic risk meters, and interactive diagnostic analytics.
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
    page_title="PneumoAI - Lung Cancer Clinical Diagnostic Engine",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced Modern Healthcare Glassmorphic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(37, 99, 235, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(147, 51, 234, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.04) 0%, transparent 60%),
                    #080C14;
        color: #F1F5F9;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.75) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #38BDF8 40%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 780px;
        line-height: 1.6;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.4rem 0.9rem;
        border-radius: 9999px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    .status-pulse {
        width: 8px;
        height: 8px;
        background-color: #34D399;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #34D399;
    }

    /* Glassmorphism Section Card Containers */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.6rem 1.6rem;
        backdrop-filter: blur(14px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 1.2rem;
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        transform: translateY(-2px);
    }

    .card-header {
        display: flex;
        align-items: center;
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .card-header span {
        font-size: 1.35rem;
        margin-right: 0.6rem;
    }

    /* Diagnosis Result Banners */
    .result-card-high {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.5) 0%, rgba(69, 10, 10, 0.8) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 15px 35px rgba(220, 38, 38, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
    }

    .result-card-mod {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.5) 0%, rgba(69, 26, 3, 0.8) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 15px 35px rgba(217, 119, 6, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
    }

    .result-card-low {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.5) 0%, rgba(2, 44, 34, 0.8) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 15px 35px rgba(5, 150, 105, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem;
    }

    /* Stat Metric Boxes */
    .stat-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }
    .stat-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.4) 0%, rgba(99, 102, 241, 0.4) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(96, 165, 250, 0.5) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Action Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%);
        color: #FFFFFF;
        font-family: 'Outfit', sans-serif;
        font-size: 1.12rem;
        font-weight: 700;
        padding: 0.85rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.45);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.6);
        border-color: rgba(255, 255, 255, 0.4);
    }

    /* Recommendation Pills */
    .rec-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 4px solid #38BDF8;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: #E2E8F0;
        display: flex;
        align-items: center;
    }
    .flag-tag {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #F87171;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
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

    # Premium Hero Banner
    st.markdown("""
    <div class="hero-container">
        <div>
            <div class="hero-title">🫁 PneumoAI Diagnostic System</div>
            <div class="hero-subtitle">
                Clinical decision support platform powered by ensemble machine learning and structured feature engineering.
                Designed for early pulmonary malignancy risk triage and precision oncology screening.
            </div>
        </div>
        <div>
            <div class="status-pill">
                <div class="status-pulse"></div>
                Engine Online (96.3% Accuracy)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔬 Individual Patient Screener",
        "📂 Cohort Batch Screening",
        "📊 Clinical Metrics & Model Explainability"
    ])

    # =========================================================================
    # TAB 1: INDIVIDUAL PATIENT ASSESSMENT
    # =========================================================================
    with tab1:
        st.markdown("<p style='color:#94A3B8; font-size:1.05rem; margin-bottom: 1.5rem;'>Enter the patient's symptomatic presentation and lifestyle risk indicators below for automated risk assessment.</p>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="glass-card">
                <div class="card-header"><span>👤</span> Demographics & Lifestyle</div>
            </div>
            """, unsafe_allow_html=True)
            gender = st.selectbox("Biological Sex", ["Male", "Female"])
            age = st.slider("Chronological Age (Years)", min_value=18, max_value=95, value=62, step=1)
            smoking = st.selectbox("Active Tobacco Smoking?", ["No", "Yes"], index=1)
            yellow_fingers = st.selectbox("Yellow Finger Nicotine Staining?", ["No", "Yes"], index=1)
            alcohol = st.selectbox("Frequent Alcohol Consumption?", ["No", "Yes"], index=1)

        with col2:
            st.markdown("""
            <div class="glass-card">
                <div class="card-header"><span>🫁</span> Respiratory Airway Symptoms</div>
            </div>
            """, unsafe_allow_html=True)
            coughing = st.selectbox("Persistent Chronic Cough?", ["No", "Yes"], index=1)
            wheezing = st.selectbox("Audible Wheezing / Stridor?", ["No", "Yes"], index=1)
            sob = st.selectbox("Shortness of Breath (Dyspnea)?", ["No", "Yes"], index=1)
            chest_pain = st.selectbox("Thoracic / Chest Pain?", ["No", "Yes"], index=1)
            swallowing = st.selectbox("Swallowing Difficulty (Dysphagia)?", ["No", "Yes"], index=1)

        with col3:
            st.markdown("""
            <div class="glass-card">
                <div class="card-header"><span>🩺</span> Systemic & Psychological</div>
            </div>
            """, unsafe_allow_html=True)
            fatigue = st.selectbox("Unexplained Fatigue / Malaise?", ["No", "Yes"], index=1)
            chronic_disease = st.selectbox("Chronic Pre-existing Illness?", ["No", "Yes"], index=1)
            allergy = st.selectbox("Known Respiratory Allergies?", ["No", "Yes"], index=0)
            anxiety = st.selectbox("Diagnosed Anxiety / Stress?", ["No", "Yes"], index=0)
            peer_pressure = st.selectbox("Environmental / Peer Smoke Exposure?", ["No", "Yes"], index=1)

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

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        assess_button = st.button("🚀 Analyze Clinical Malignancy Probability", use_container_width=True)

        if assess_button:
            result = predictor.predict_single(patient_payload)
            prob = result["risk_probability"]
            prob_percent = prob * 100.0
            tier = result["risk_tier"]

            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-family: Outfit; font-weight:700;'>📊 Diagnostic Risk Assessment Outcome</h3>", unsafe_allow_html=True)

            res_col1, res_col2 = st.columns([1.1, 0.9])

            with res_col1:
                if prob >= 0.70:
                    st.markdown(f"""
                    <div class="result-card-high">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="color:#FCA5A5; margin:0; font-family: Outfit; font-size:1.5rem;">⚠️ {tier}</h3>
                            <span style="background:rgba(239,68,68,0.25); border:1px solid #EF4444; color:#FCA5A5; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.85rem;">CRITICAL</span>
                        </div>
                        <p style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; font-family: Outfit; margin: 0.8rem 0 0.4rem 0;">
                            {prob_percent:.1f}%
                        </p>
                        <p style="color: #FECACA; font-size: 0.95rem; line-height: 1.5;">
                            High statistical concordance with active pulmonary neoplastic presentation. Immediate secondary oncology diagnostic workup is strongly recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown(f"""
                    <div class="result-card-mod">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="color:#FDE68A; margin:0; font-family: Outfit; font-size:1.5rem;">⚡ {tier}</h3>
                            <span style="background:rgba(245,158,11,0.25); border:1px solid #F59E0B; color:#FDE68A; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.85rem;">MODERATE</span>
                        </div>
                        <p style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; font-family: Outfit; margin: 0.8rem 0 0.4rem 0;">
                            {prob_percent:.1f}%
                        </p>
                        <p style="color: #FEF3C7; font-size: 0.95rem; line-height: 1.5;">
                            Borderline risk markers detected. Intermediate pulmonary symptom burden warrants surveillance screening or follow-up spirometry.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-low">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="color:#A7F3D0; margin:0; font-family: Outfit; font-size:1.5rem;">✅ {tier}</h3>
                            <span style="background:rgba(16,185,129,0.25); border:1px solid #10B981; color:#A7F3D0; padding:4px 10px; border-radius:6px; font-weight:700; font-size:0.85rem;">LOW RISK</span>
                        </div>
                        <p style="font-size: 2.4rem; font-weight: 800; color: #FFFFFF; font-family: Outfit; margin: 0.8rem 0 0.4rem 0;">
                            {prob_percent:.1f}%
                        </p>
                        <p style="color: #D1FAE5; font-size: 0.95rem; line-height: 1.5;">
                            Patient profile demonstrates low concordance with primary clinical indicators of malignant lung disease. Standard annual wellness checks advised.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.progress(prob)

            with res_col2:
                st.markdown("<h4 style='font-family: Outfit; color: #E2E8F0; margin-top:0;'>🩺 Clinical Recommendations</h4>", unsafe_allow_html=True)
                for rec in result["clinical_guidance"]:
                    st.markdown(f"<div class='rec-item'>👉 {rec}</div>", unsafe_allow_html=True)

                if result["reported_symptoms"]:
                    st.markdown("<h4 style='font-family: Outfit; color: #E2E8F0; margin-top:1.2rem;'>🚩 Active Clinical Risk Flags</h4>", unsafe_allow_html=True)
                    flags_html = "".join([f"<span class='flag-tag'>{s.replace('_', ' ').title()}</span>" for s in result["reported_symptoms"]])
                    st.markdown(flags_html, unsafe_allow_html=True)

            # Exportable Clinical Report
            report_text = f"""=====================================================
PNEUMO-AI CLINICAL DECISION SUPPORT REPORT
Diagnostic Machine Learning Assessment
=====================================================
Patient Demographics:
  Biological Sex: {gender}
  Age: {age} years

Assessment Summary:
  Diagnostic Prediction: {result['prediction_label']}
  Calculated Malignancy Probability: {prob_percent:.2f}%
  Stratification Tier: {tier}

Reported Risk Factors:
  {', '.join([s.replace('_', ' ').title() for s in result['reported_symptoms']]) if result['reported_symptoms'] else 'None reported'}

Recommended Clinical Next Steps:
""" + "\n".join([f"  * {r}" for r in result["clinical_guidance"]]) + """

Notice: This computerized output provides risk-stratification decision guidance
and does not substitute for pathological histology or tissue biopsy.
=====================================================
"""
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Export Certified Clinical Decision Report (.txt)",
                data=report_text,
                file_name=f"LungCancer_Report_{gender}_Age{age}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # =========================================================================
    # TAB 2: BATCH SCREENING
    # =========================================================================
    with tab2:
        st.markdown("<p style='color:#94A3B8; font-size:1.05rem;'>Upload patient cohort spreadsheets for automated high-throughput screening.</p>", unsafe_allow_html=True)

        sample_csv_path = os.path.join(project_root, "data", "raw", "survey_lung_cancer.csv")
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="📄 Download Clinical Cohort CSV Template",
                    data=f.read(),
                    file_name="clinical_cohort_template.csv",
                    mime="text/csv"
                )

        uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully ingested cohort of **{len(batch_df)}** clinical records.")

            with st.spinner("Executing ensemble predictive pipeline..."):
                scored_df = predictor.predict_batch(batch_df)

            st.dataframe(scored_df.head(20), use_container_width=True)

            high_count = (scored_df["RISK_TIER"].str.contains("High")).sum()
            mod_count = (scored_df["RISK_TIER"].str.contains("Moderate")).sum()
            low_count = (scored_df["RISK_TIER"].str.contains("Low")).sum()

            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-val" style="color:#F87171;">{high_count}</div>
                    <div class="stat-label">High Risk Cases ({high_count/len(scored_df)*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with b2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-val" style="color:#FBBF24;">{mod_count}</div>
                    <div class="stat-label">Moderate Risk Cases ({mod_count/len(scored_df)*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with b3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-val" style="color:#34D399;">{low_count}</div>
                    <div class="stat-label">Low Risk Cases ({low_count/len(scored_df)*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)

            csv_export = scored_df.to_csv(index=False).encode("utf-8")
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Export Scored Patient Cohort (CSV)",
                data=csv_export,
                file_name="scored_lung_cancer_predictions.csv",
                mime="text/csv"
            )

    # =========================================================================
    # TAB 3: MODEL PERFORMANCE & EXPLAINABILITY
    # =========================================================================
    with tab3:
        st.markdown("<p style='color:#94A3B8; font-size:1.05rem;'>Validated on unseen clinical patient records using 5-Fold Stratified Cross-Validation.</p>", unsafe_allow_html=True)

        if metrics:
            fm = metrics.get("final_metrics", {})
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""<div class="stat-box"><div class="stat-val">{fm.get('accuracy', 0)*100:.1f}%</div><div class="stat-label">Accuracy</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="stat-box"><div class="stat-val">{fm.get('sensitivity_recall', 0)*100:.1f}%</div><div class="stat-label">Sensitivity (Recall)</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="stat-box"><div class="stat-val">{fm.get('specificity', 0)*100:.1f}%</div><div class="stat-label">Specificity</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="stat-box"><div class="stat-val">{fm.get('precision', 0)*100:.1f}%</div><div class="stat-label">Precision</div></div>""", unsafe_allow_html=True)
            with m5:
                st.markdown(f"""<div class="stat-box"><div class="stat-val">{fm.get('roc_auc', 0):.3f}</div><div class="stat-label">ROC-AUC Score</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-family: Outfit; font-weight:700;'>📈 Publication-Ready Diagnostic Visualizations</h3>", unsafe_allow_html=True)

        fig_col1, fig_col2 = st.columns(2)
        reports_dir = os.path.join(project_root, "reports", "figures")
        cm_file = os.path.join(reports_dir, "confusion_matrix.png")
        roc_file = os.path.join(reports_dir, "roc_curve.png")
        feat_file = os.path.join(reports_dir, "feature_importance.png")
        comp_file = os.path.join(reports_dir, "model_comparison.png")

        with fig_col1:
            if os.path.exists(cm_file):
                st.image(cm_file, caption="Confusion Matrix on Test Patients", use_container_width=True)
            if os.path.exists(feat_file):
                st.image(feat_file, caption="Key Contributing Predictive Symptoms", use_container_width=True)

        with fig_col2:
            if os.path.exists(roc_file):
                st.image(roc_file, caption="Receiver Operating Characteristic (ROC Curve)", use_container_width=True)
            if os.path.exists(comp_file):
                st.image(comp_file, caption="Benchmark Comparison Across All 5 ML Algorithms", use_container_width=True)


if __name__ == "__main__":
    main()
