"""
Bank Marketing Term Deposit Prediction App
============================================
Predicts whether a bank client will subscribe to a term deposit
based on demographic, campaign, and economic data.

Author: Davian
Course: CAI2C08 - Machine Learning for Developers
Institution: Temasek Polytechnic
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
    initial_sidebar_state="expanded"
)

# Custom CSS with DARK VISIBLE LABELS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Main dark background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* =====================================================
       CRITICAL: Make ALL widget labels DARK and VISIBLE
       These labels appear above input fields
       ===================================================== */
    
    /* Target ALL possible label elements */
    label {
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-shadow: none !important;
        background: transparent !important;
    }
    
    /* Streamlit specific label selectors */
    .stSelectbox > label,
    .stSlider > label,
    .stNumberInput > label,
    .stTextInput > label,
    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] {
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    
    /* Force dark text on any label-like element */
    .stSelectbox label p,
    .stSlider label p,
    .stNumberInput label p {
        color: #1a1a2e !important;
    }
    
    /* Section titles h3 (Demographics, Financial Status) - WHITE on dark bg */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Input field styling - DARK boxes with WHITE text */
    .stSelectbox > div > div,
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 8px !important;
    }
    
    /* Text inside dropdowns - WHITE */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #ffffff !important;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 2px solid #4a5568 !important;
    }
    
    input[type="number"] {
        color: #ffffff !important;
    }
    
    /* Slider text */
    .stSlider > div > div > div {
        color: #1a1a2e !important;
    }
    
    /* Section cards - Light background */
    .section-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 15px 20px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border-left: 5px solid #667eea;
    }
    
    .section-card h4 {
        color: #1a1a2e !important;
        margin: 0;
        font-weight: 700;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1, .main-header p {
        color: white !important;
    }
    
    /* Result boxes */
    .result-yes {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(17, 153, 142, 0.4);
        margin: 20px 0;
    }
    
    .result-yes * { color: white !important; }
    
    .result-no {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(235, 51, 73, 0.4);
        margin: 20px 0;
    }
    
    .result-no * { color: white !important; }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 18px 35px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #38ef7d !important;
        font-size: 24px !important;
    }
    
    /* Analysis text - WHITE */
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found!")
        return None


def create_feature_dataframe(inputs):
    """Create DataFrame with feature engineering."""
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
        <h1 style="margin:0; font-size:2.5em;">🏦 Bank Marketing Predictor</h1>
        <p style="font-size:1.2em; margin-top:10px;">Predict Term Deposit Subscription Likelihood</p>
    </div>
    """, unsafe_allow_html=True)
    
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 About")
        st.info("Predicts if a client will subscribe to a term deposit.")
        
        st.markdown("---")
        st.markdown("## 🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "87.0%")
        with col2:
            st.metric("F1 Score", "29.8%")
        
        st.markdown("---")
        st.markdown("## 📈 Key Predictors")
        st.markdown("""
        - ✅ Previous Success
        - 📊 Economic Indicators
        - 👤 Age & Job Type
        - 📱 Contact Method
        """)
        
        st.markdown("---")
        st.caption("CAI2C08 - ML for Developers")
        st.caption("Temasek Polytechnic © 2025")
    
    # ========== CUSTOMER INFORMATION ==========
    st.markdown('<div class="section-card"><h4>📝 Customer Information</h4></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        age = st.slider("Age", min_value=18, max_value=95, value=35)
        job = st.selectbox("Job Type", options=[
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician',
            'unemployed', 'unknown'
        ])
        marital = st.selectbox("Marital Status", options=['single', 'married', 'divorced'])
        education = st.selectbox("Education Level", options=[
            'basic.4y', 'basic.6y', 'basic.9y', 'high.school',
            'illiterate', 'professional.course', 'university.degree', 'unknown'
        ], index=6)
    
    with col2:
        st.markdown("### 💳 Financial Status")
        default = st.selectbox("Credit in Default?", options=['no', 'yes', 'unknown'])
        housing = st.selectbox("Housing Loan?", options=['no', 'yes', 'unknown'])
        loan = st.selectbox("Personal Loan?", options=['no', 'yes', 'unknown'])
    
    # ========== CAMPAIGN INFORMATION ==========
    st.markdown('<div class="section-card"><h4>📞 Campaign Information</h4></div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 📱 Contact Details")
        contact = st.selectbox("Contact Method", options=['cellular', 'telephone'])
        month = st.selectbox("Last Contact Month", options=[
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ], index=4)
        day_of_week = st.selectbox("Day of Week", options=['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Contacts This Campaign", min_value=1, max_value=56, value=2)
    
    with col4:
        st.markdown("### 📜 Previous Campaign")
        poutcome = st.selectbox("Previous Outcome", options=['nonexistent', 'failure', 'success'])
        pdays = st.number_input("Days Since Last Contact", min_value=0, max_value=999, value=999,
                                help="999 = never contacted before")
        previous = st.number_input("Previous Contacts", min_value=0, max_value=7, value=0)
    
    # ========== ECONOMIC INDICATORS ==========
    st.markdown('<div class="section-card"><h4>📊 Economic Indicators</h4></div>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        emp_var_rate = st.selectbox("Employment Variation Rate", 
                                     options=[-3.4, -2.9, -1.8, -1.1, -0.1, 1.1, 1.4], index=3)
        cons_price_idx = st.slider("Consumer Price Index", 
                                    min_value=92.0, max_value=95.0, value=93.5, step=0.1)
        cons_conf_idx = st.slider("Consumer Confidence Index",
                                   min_value=-51.0, max_value=-26.0, value=-40.0, step=0.5)
    
    with col6:
        euribor3m = st.slider("Euribor 3-Month Rate",
                               min_value=0.5, max_value=5.1, value=2.5, step=0.1)
        nr_employed = st.selectbox("Number of Employees",
                                    options=[4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 5176.3, 5191.0, 5195.8, 5228.1],
                                    index=4)
    
    # ========== PREDICT BUTTON ==========
    st.markdown("---")
    
    if st.button("🔮 PREDICT SUBSCRIPTION", use_container_width=True):
        inputs = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed
        }
        
        input_df = create_feature_dataframe(inputs)
        feature_cols = model_package['feature_columns']
        input_df = input_df[feature_cols]
        
        try:
            X_processed = model_package['preprocessor'].transform(input_df)
            prediction = model_package['model'].predict(X_processed)[0]
            probability = model_package['model'].predict_proba(X_processed)[0]
            
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = probability[prediction] * 100
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-yes">
                        <h2 style="margin:0; font-size:1.8em;">✅ LIKELY TO SUBSCRIBE</h2>
                        <p style="font-size:1.8em; margin:15px 0; font-weight:700;">{confidence:.1f}% Confidence</p>
                        <p style="margin:0;">High potential for subscription!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no">
                        <h2 style="margin:0; font-size:1.8em;">❌ UNLIKELY TO SUBSCRIBE</h2>
                        <p style="font-size:1.8em; margin:15px 0; font-weight:700;">{confidence:.1f}% Confidence</p>
                        <p style="margin:0;">May need additional engagement.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("### 📊 Probability")
                st.markdown(f"**Subscribe:** {probability[1]*100:.1f}%")
                st.progress(probability[1])
                st.markdown(f"**Not Subscribe:** {probability[0]*100:.1f}%")
                st.progress(probability[0])
            
            # Key Factors
            st.markdown("### 💡 Key Factors")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if poutcome == 'success':
                    st.markdown("✅ **Previous Success** - Very high potential!")
                if emp_var_rate < 0:
                    st.markdown("✅ **Negative Employment Rate** - Better timing")
                if contact == 'cellular':
                    st.markdown("✅ **Cellular Contact** - More effective")
                if job in ['student', 'retired']:
                    st.markdown("✅ **Job Type** - Higher subscription rates")
            
            with col_b:
                if poutcome == 'failure':
                    st.markdown("⚠️ **Previous Failure** - Try different approach")
                if campaign > 5:
                    st.markdown("⚠️ **Many Contacts** - Risk of fatigue")
                if cons_conf_idx < -45:
                    st.markdown("⚠️ **Low Consumer Confidence**")
        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:rgba(255,255,255,0.6); padding:20px;">
        <p>🏦 Bank Marketing Prediction System | CAI2C08 | Temasek Polytechnic</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
