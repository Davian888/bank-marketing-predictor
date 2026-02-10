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
    layout="wide"
)

# Force light theme and clean styling
st.markdown("""
<style>
    /* Force light background */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1a365d 0%, #2c5282 100%);
        padding: 30px 40px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        color: #a0aec0 !important;
        font-size: 16px;
        margin-top: 8px;
    }
    
    /* Section headers */
    .section-header {
        color: #1a365d !important;
        font-size: 20px;
        font-weight: 600;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #3182ce;
    }
    
    /* Metrics styling */
    .metric-container {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-value {
        color: #2b6cb0 !important;
        font-size: 28px;
        font-weight: 700;
    }
    
    .metric-label {
        color: #4a5568 !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input labels - IMPORTANT for visibility */
    .stSelectbox > label, .stSlider > label, .stNumberInput > label,
    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Make all text dark for visibility */
    p, span, label, .stMarkdown {
        color: #2d3748 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a365d !important;
    }
    
    /* Input boxes */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #ffffff !important;
        color: #2d3748 !important;
        border: 1px solid #cbd5e0 !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #2b6cb0 0%, #3182ce 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2c5282 0%, #2b6cb0 100%) !important;
    }
    
    /* Result boxes */
    .result-box-yes {
        background: linear-gradient(135deg, #276749 0%, #38a169 100%);
        color: white !important;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    
    .result-box-no {
        background: linear-gradient(135deg, #c53030 0%, #e53e3e 100%);
        color: white !important;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    
    .result-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
        color: white !important;
    }
    
    .result-confidence {
        font-size: 18px;
        opacity: 0.95;
        color: white !important;
    }
    
    /* Cards for input sections */
    .input-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .card-title {
        color: #2d3748 !important;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #3182ce;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #2d3748 !important;
        font-weight: 600 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #3182ce !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'bank_marketing_model.pkl' is in the same directory.")
        return None


def create_features(inputs):
    """Feature engineering matching training pipeline."""
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
    # ==================== HEADER ====================
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🏦 Bank Marketing Predictor</div>
        <div class="header-subtitle">Predict Term Deposit Subscription using Machine Learning | No Duration (Data Leakage Removed)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    # Get metrics
    metrics = model_package.get('metrics', {})
    threshold = model_package.get('threshold', 0.5)
    
    # ==================== MODEL METRICS ====================
    st.markdown('<div class="section-header">📊 Model Performance</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics.get('accuracy', 'N/A')}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics.get('f1_score', 'N/A')}%</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics.get('precision', 'N/A')}%</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics.get('recall', 'N/A')}%</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{metrics.get('roc_auc', 'N/A')}</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== CLIENT INFORMATION ====================
    st.markdown('<div class="section-header">📝 Client Information</div>', unsafe_allow_html=True)
    
    # Row 1: Demographics
    st.markdown("**👤 Personal Details**")
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
    
    # Row 2: Financial
    st.markdown("**💳 Financial Status**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
    with col2:
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    with col3:
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    # Row 3: Campaign
    st.markdown("**📞 Campaign Information**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
    with col2:
        month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
    with col3:
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
    with col4:
        campaign = st.number_input("Contacts This Campaign", 1, 50, 2)
    
    # Row 4: Previous Campaign
    st.markdown("**📜 Previous Campaign**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pdays = st.number_input("Days Since Last Contact (999=never)", 0, 999, 999)
    with col2:
        previous = st.number_input("Previous Contacts", 0, 50, 0)
    with col3:
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    # Row 5: Economic Indicators
    st.markdown("**📈 Economic Indicators**")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        emp_var_rate = st.number_input("Emp. Var. Rate", -3.5, 1.5, -0.1, 0.1)
    with col2:
        cons_price_idx = st.number_input("Consumer Price Idx", 92.0, 95.0, 93.5, 0.1)
    with col3:
        cons_conf_idx = st.number_input("Consumer Conf. Idx", -51.0, -26.0, -40.0, 0.5)
    with col4:
        euribor3m = st.number_input("Euribor 3M Rate", 0.5, 5.1, 2.5, 0.1)
    with col5:
        nr_employed = st.selectbox("Nr. Employed (k)", 
                                    [4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 
                                     5176.3, 5191.0, 5195.8, 5228.1], index=4)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==================== PREDICT BUTTON ====================
    if st.button("🔮 PREDICT SUBSCRIPTION LIKELIHOOD"):
        inputs = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
            'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
            'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }
        
        # Create features
        input_df = create_features(inputs)
        input_df = input_df[model_package['feature_columns']]
        
        try:
            # Predict
            X_processed = model_package['preprocessor'].transform(input_df)
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            threshold = model_package.get('threshold', 0.5)
            prediction = 1 if proba[1] >= threshold else 0
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = proba[prediction] * 100
            
            # ==================== RESULTS ====================
            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-box-yes">
                        <div class="result-title">✅ LIKELY TO SUBSCRIBE</div>
                        <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-no">
                        <div class="result-title">❌ UNLIKELY TO SUBSCRIBE</div>
                        <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Probability Breakdown**")
                st.write(f"Subscribe (Yes): **{proba[1]*100:.1f}%**")
                st.progress(proba[1])
                st.write(f"Not Subscribe (No): **{proba[0]*100:.1f}%**")
                st.progress(proba[0])
            
            # Insights
            st.markdown("**💡 Key Insights**")
            col1, col2 = st.columns(2)
            
            with col1:
                if poutcome == 'success':
                    st.success("✓ Previous campaign success - strong indicator!")
                if contact == 'cellular':
                    st.info("✓ Cellular contact tends to be more effective")
                if emp_var_rate < 0:
                    st.info("✓ Lower employment variation rate is favorable")
            
            with col2:
                if campaign > 5:
                    st.warning("⚠ High contact frequency may indicate resistance")
                if pdays == 999:
                    st.info("ℹ Client was not previously contacted")
                if cons_conf_idx < -45:
                    st.warning("⚠ Low consumer confidence may affect decision")
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 14px;">
        <strong>Bank Marketing Prediction System</strong><br>
        CAI2C08 - Machine Learning for Developers | Temasek Polytechnic<br>
        Dataset: UCI Bank Marketing (Economic Indicators Included, Duration Removed)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
