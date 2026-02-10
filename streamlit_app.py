"""
Bank Marketing Term Deposit Prediction App
Author: Davian | CAI2C08 - Machine Learning for Developers | Temasek Polytechnic
Dataset: UCI Bank Marketing Dataset (with Economic Indicators, Duration Removed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    /* Light background */
    .stApp {
        background: linear-gradient(180deg, #f0f4f8 0%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .header-box {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        color: #a0c4e8 !important;
        font-size: 14px;
        margin-top: 8px;
    }
    
    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    
    .metric-value {
        color: #2b6cb0 !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        color: #64748b !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }
    
    /* Section headers */
    .section-header {
        color: #1a365d !important;
        font-size: 18px;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #3182ce;
    }
    
    /* Subsection headers */
    .subsection {
        color: #2d3748 !important;
        font-size: 14px;
        font-weight: 600;
        margin: 15px 0 10px 0;
    }
    
    /* All text dark for visibility */
    p, span, label, div, .stMarkdown p {
        color: #2d3748 !important;
    }
    
    h1, h2, h3 {
        color: #1a365d !important;
    }
    
    /* Result boxes */
    .result-yes {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }
    
    .result-no {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(239,68,68,0.3);
    }
    
    .result-title {
        color: #ffffff !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    
    .result-conf {
        color: #ffffff !important;
        font-size: 16px;
        margin-top: 8px;
        opacity: 0.95;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #3182ce 100%) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(43,108,176,0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%) !important;
    }
    
    /* Probability box */
    .prob-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b !important;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'bank_marketing_model.pkl' not found!")
        return None


# ===================== FEATURE ENGINEERING =====================
def create_features(inputs):
    """Create engineered features matching training pipeline."""
    df = pd.DataFrame([inputs])
    
    # Engineered features
    df['was_contacted_before'] = (df['previous'] > 0).astype(int)
    df['has_loan_burden'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
    df['prev_success'] = (df['poutcome'] == 'success').astype(int)
    
    def age_group(age):
        if age < 25: return 'young'
        elif age < 35: return 'young_adult'
        elif age < 50: return 'middle_age'
        elif age < 65: return 'senior'
        else: return 'elderly'
    
    df['age_group'] = df['age'].apply(age_group)
    df['economic_sentiment'] = (df['cons.conf.idx'] - (-50.8)) / ((-26.9) - (-50.8))
    
    return df


# ===================== MAIN APP =====================
def main():
    # ----- HEADER -----
    st.markdown("""
    <div class="header-box">
        <div class="header-title">🏦 Bank Marketing Term Deposit Predictor</div>
        <div class="header-subtitle">UCI Dataset | Economic Indicators Included | Duration Removed (No Data Leakage)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    metrics = model_package.get('metrics', {})
    threshold = model_package.get('threshold', 0.5)
    
    # ----- MODEL METRICS -----
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("accuracy", "N/A")}%</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("f1_score", "N/A")}%</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("precision", "N/A")}%</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("recall", "N/A")}%</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("roc_auc", "N/A")}</div><div class="metric-label">ROC-AUC</div></div>', unsafe_allow_html=True)
    
    # ----- CLIENT INFORMATION -----
    st.markdown('<div class="section-header">📝 Client Information</div>', unsafe_allow_html=True)
    
    # Row 1: Personal Details
    st.markdown('<div class="subsection">👤 Personal Details</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.slider("Age", 18, 95, 35)
    with col2:
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                    'management', 'retired', 'self-employed', 'services', 
                                    'student', 'technician', 'unemployed', 'unknown'])
    with col3:
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
    with col4:
        education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                'illiterate', 'professional.course', 
                                                'university.degree', 'unknown'])
    
    # Row 2: Financial Status
    st.markdown('<div class="subsection">💳 Financial Status</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
    with col2:
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    with col3:
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    # Row 3: Campaign Info
    st.markdown('<div class="subsection">📞 Current Campaign</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    with col2:
        month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
    with col3:
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    with col4:
        campaign = st.slider("Contacts (This Campaign)", 1, 50, 2)
    
    # Row 4: Previous Campaign
    st.markdown('<div class="subsection">📜 Previous Campaign</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pdays = st.selectbox("Days Since Last Contact", 
                              [999, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 90, 180],
                              help="999 means client was never contacted before")
    with col2:
        previous = st.slider("Previous Contacts", 0, 10, 0)
    with col3:
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    # Row 5: Economic Indicators
    st.markdown('<div class="subsection">📈 Economic Indicators</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        emp_var_rate = st.slider("Emp. Var. Rate", -3.5, 1.5, -0.1, 0.1)
    with col2:
        cons_price_idx = st.slider("Consumer Price Idx", 92.0, 95.0, 93.5, 0.1)
    with col3:
        cons_conf_idx = st.slider("Consumer Conf. Idx", -51.0, -26.0, -40.0, 0.5)
    with col4:
        euribor3m = st.slider("Euribor 3M Rate", 0.5, 5.1, 2.5, 0.1)
    with col5:
        nr_employed = st.selectbox("Nr. Employed (k)", 
                                    [4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 
                                     5176.3, 5191.0, 5195.8, 5228.1], index=4)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ----- PREDICT BUTTON -----
    if st.button("🔮 PREDICT SUBSCRIPTION LIKELIHOOD"):
        inputs = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
            'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
            'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }
        
        try:
            # Feature engineering
            input_df = create_features(inputs)
            input_df = input_df[model_package['feature_columns']]
            
            # Prediction
            X_processed = model_package['preprocessor'].transform(input_df)
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            prediction = 1 if proba[1] >= threshold else 0
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = proba[prediction] * 100
            
            # ----- RESULTS -----
            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-yes">
                        <div class="result-title">✅ LIKELY TO SUBSCRIBE</div>
                        <div class="result-conf">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no">
                        <div class="result-title">❌ UNLIKELY TO SUBSCRIBE</div>
                        <div class="result-conf">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="prob-box">', unsafe_allow_html=True)
                st.markdown("**Probability**")
                st.write(f"Subscribe: **{proba[1]*100:.1f}%**")
                st.progress(proba[1])
                st.write(f"Not Subscribe: **{proba[0]*100:.1f}%**")
                st.progress(proba[0])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Insights
            with st.expander("💡 Prediction Insights"):
                if poutcome == 'success':
                    st.success("✓ Previous campaign success strongly predicts subscription")
                if contact == 'cellular':
                    st.info("✓ Cellular contact is generally more effective")
                if emp_var_rate < 0:
                    st.info("✓ Negative employment variation correlates with higher subscription")
                if campaign > 5:
                    st.warning("⚠ High number of contacts may indicate client resistance")
                if pdays == 999:
                    st.info("ℹ Client was not previously contacted")
        
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")
    
    # ----- FOOTER -----
    st.markdown("""
    <div class="footer-text">
        <strong>Bank Marketing Term Deposit Prediction System</strong><br>
        CAI2C08 - Machine Learning for Developers | Temasek Polytechnic<br>
        Model: Gradient Boosting with SMOTE | Threshold: {:.2f}
    </div>
    """.format(threshold), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
