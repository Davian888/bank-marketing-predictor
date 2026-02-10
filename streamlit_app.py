"""
Bank Marketing Term Deposit Prediction App
Author: Davian | CAI2C08 - Machine Learning for Developers | Temasek Polytechnic
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean Minimalistic CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean header */
    .main-header {
        background: #ffffff;
        padding: 2rem 3rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 2rem;
        border-left: 4px solid #2563eb;
    }
    
    .main-header h1 {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: #64748b;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cards */
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .card-title {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    /* Metrics */
    .metrics-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .metric-box {
        text-align: center;
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Result boxes */
    .result-yes {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .result-no {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .result-yes h2, .result-no h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }
    
    .result-yes p, .result-no p {
        margin: 0;
        opacity: 0.9;
    }
    
    /* Button */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        border-radius: 8px;
        width: 100%;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
    }
    
    /* Input labels */
    label, .stSelectbox label, .stSlider label, .stNumberInput label,
    div[data-testid="stWidgetLabel"] {
        color: #374151 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        color: #2563eb;
    }
    
    /* Section divider */
    .section-title {
        color: #1e293b;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2563eb;
        display: inline-block;
    }
    
    /* Info text */
    .info-text {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("Model file not found!")
        return None


def create_features(inputs):
    """Feature engineering matching training."""
    df = pd.DataFrame([inputs])
    
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


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bank Marketing Predictor</h1>
        <p>Predict term deposit subscription likelihood using machine learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    # Get metrics
    metrics = model_package.get('metrics', {})
    threshold = model_package.get('threshold', 0.5)
    
    # Layout: Metrics bar
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}%")
    m2.metric("F1 Score", f"{metrics.get('f1_score', 'N/A')}%")
    m3.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
    m4.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
    m5.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
    
    st.markdown("---")
    
    # Input sections
    st.markdown('<p class="section-title">Client Information</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Demographics & Financial", "📞 Campaign Details", "📊 Economic Indicators"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Personal Details**")
            age = st.slider("Age", 18, 95, 35)
            job = st.selectbox("Occupation", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                               'management', 'retired', 'self-employed', 'services', 
                                               'student', 'technician', 'unemployed', 'unknown'])
            marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
            education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                    'illiterate', 'professional.course', 
                                                    'university.degree', 'unknown'])
        
        with col2:
            st.markdown("**Financial Status**")
            default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])
            housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
            loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Campaign**")
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
            month = st.selectbox("Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
            day_of_week = st.selectbox("Contact Day", ['mon', 'tue', 'wed', 'thu', 'fri'])
            campaign = st.number_input("Number of Contacts", 1, 50, 2)
        
        with col2:
            st.markdown("**Previous Campaign**")
            pdays = st.number_input("Days Since Last Contact", -1, 999, 999, 
                                    help="999 = not previously contacted")
            previous = st.number_input("Previous Campaign Contacts", 0, 50, 0)
            poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    with tab3:
        st.markdown("**Economic Context** — These indicators represent market conditions during the campaign")
        
        col1, col2 = st.columns(2)
        
        with col1:
            emp_var_rate = st.slider("Employment Variation Rate", -3.5, 1.5, -0.1, 0.1)
            cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.5, 0.1)
            cons_conf_idx = st.slider("Consumer Confidence Index", -51.0, -26.0, -40.0, 0.5)
        
        with col2:
            euribor3m = st.slider("Euribor 3-Month Rate", 0.5, 5.1, 2.5, 0.1)
            nr_employed = st.selectbox("Number Employed (thousands)", 
                                        [4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 
                                         5176.3, 5191.0, 5195.8, 5228.1], index=4)
    
    # Predict button
    st.markdown("---")
    
    if st.button("🔍 Predict Subscription Likelihood"):
        inputs = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
            'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
            'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }
        
        input_df = create_features(inputs)
        input_df = input_df[model_package['feature_columns']]
        
        try:
            X_processed = model_package['preprocessor'].transform(input_df)
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            threshold = model_package.get('threshold', 0.5)
            prediction = 1 if proba[1] >= threshold else 0
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = proba[prediction] * 100
            
            # Results
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-yes">
                        <h2>✓ Likely to Subscribe</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no">
                        <h2>✗ Unlikely to Subscribe</h2>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Probability**")
                st.progress(proba[1])
                st.caption(f"Subscribe: {proba[1]*100:.1f}% | Not Subscribe: {proba[0]*100:.1f}%")
            
            # Quick insights
            with st.expander("📊 Quick Insights"):
                insights = []
                if poutcome == 'success':
                    insights.append("✓ Previous campaign success increases likelihood")
                if emp_var_rate < 0:
                    insights.append("✓ Negative employment rate correlates with higher subscription")
                if contact == 'cellular':
                    insights.append("✓ Cellular contact tends to be more effective")
                if campaign > 5:
                    insights.append("⚠ High number of contacts may indicate resistance")
                
                for insight in insights:
                    st.markdown(insight)
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.caption("CAI2C08 - Machine Learning for Developers | Temasek Polytechnic | Davian")


if __name__ == "__main__":
    main()
