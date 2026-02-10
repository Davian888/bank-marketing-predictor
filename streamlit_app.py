"""
Bank Marketing Term Deposit Prediction App
============================================
A machine learning application to predict customer subscription likelihood
for bank term deposits based on demographic, campaign, and economic data.

Author: Davian
Course: CAI2C08 - Machine Learning for Developers
Institution: Temasek Polytechnic

Dataset: UCI Bank Marketing Dataset (20 columns)
Note: Duration column removed to prevent data leakage
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

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    /* Label visibility fix */
    label, .stSelectbox label, .stSlider label, .stNumberInput label,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] {
        color: #1e3c72 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    .result-yes {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
    
    .result-no {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model package."""
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found!")
        return None


def create_features(inputs):
    """Create DataFrame with feature engineering matching training."""
    df = pd.DataFrame([inputs])
    
    # Feature engineering - MUST MATCH TRAINING
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
    
    # Economic sentiment (normalized using training data range)
    # cons.conf.idx range: -50.8 to -26.9
    df['economic_sentiment'] = (df['cons.conf.idx'] - (-50.8)) / ((-26.9) - (-50.8))
    
    return df


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bank Marketing Predictor</h1>
        <p style="font-size: 1.2em;">Predict Term Deposit Subscription Likelihood</p>
        <p style="font-size: 0.9em; opacity: 0.8;">With Economic Indicators | No Duration (Data Leakage Removed)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    # =========================================
    # DYNAMIC METRICS FROM PKL FILE
    # =========================================
    metrics = model_package.get('metrics', {})
    threshold = model_package.get('threshold', 0.5)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About This Tool")
        st.info("Predicts term deposit subscription using demographic, campaign, and economic data.")
        
        st.markdown("### 🎯 Model Performance")
        st.caption("*Loaded dynamically from model file*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}%")
            st.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
        with col2:
            st.metric("F1 Score", f"{metrics.get('f1_score', 'N/A')}%")
            st.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
        
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
        st.caption(f"Threshold: {threshold}")
        
        st.markdown("### 📈 Key Features")
        st.markdown("""
        - Economic indicators (emp.var.rate, cons.price.idx, etc.)
        - Previous campaign outcome
        - Contact method & timing
        - Client demographics
        """)
        
        st.markdown("### 📝 Dataset")
        st.markdown("""
        **Source:** UCI Bank Marketing  
        **Columns:** 20 (no duration)  
        **Model:** Gradient Boosting
        """)
    
    # Main Content - 3 columns for inputs
    st.markdown("### 📝 Client Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Demographics")
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                    'management', 'retired', 'self-employed', 'services', 
                                    'student', 'technician', 'unemployed', 'unknown'])
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
        education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                'illiterate', 'professional.course', 
                                                'university.degree', 'unknown'])
        
        st.markdown("#### 💳 Financial")
        default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    with col2:
        st.markdown("#### 📞 Campaign Info")
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
        day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        campaign = st.number_input("Contacts This Campaign", 1, 50, 2)
        
        st.markdown("#### 📜 Previous Campaign")
        pdays = st.number_input("Days Since Last Contact", -1, 999, 999, 
                                help="999 = not previously contacted")
        previous = st.number_input("Previous Contacts", 0, 50, 0)
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    with col3:
        st.markdown("#### 📊 Economic Indicators")
        st.caption("These represent economic conditions during the campaign")
        
        emp_var_rate = st.slider("Employment Variation Rate", -3.5, 1.5, -0.1, 0.1,
                                  help="Quarterly employment variation rate")
        cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.5, 0.1,
                                    help="Monthly consumer price index")
        cons_conf_idx = st.slider("Consumer Confidence Index", -51.0, -26.0, -40.0, 0.5,
                                   help="Monthly consumer confidence index")
        euribor3m = st.slider("Euribor 3-Month Rate", 0.5, 5.1, 2.5, 0.1,
                               help="Daily euribor 3 month rate")
        nr_employed = st.selectbox("Number Employed (thousands)", 
                                    [4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 
                                     5176.3, 5191.0, 5195.8, 5228.1],
                                    index=4,
                                    help="Quarterly indicator")
    
    # Prediction
    st.markdown("---")
    
    if st.button("🔮 Predict Subscription Likelihood", use_container_width=True):
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
        
        # Create features
        input_df = create_features(inputs)
        
        # Reorder columns to match training
        feature_cols = model_package['feature_columns']
        input_df = input_df[feature_cols]
        
        try:
            # Preprocess
            X_processed = model_package['preprocessor'].transform(input_df)
            
            # Get probability
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            # Apply threshold
            threshold = model_package.get('threshold', 0.5)
            prediction = 1 if proba[1] >= threshold else 0
            
            # Decode
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = proba[prediction] * 100
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-yes">
                        <h2>✅ LIKELY TO SUBSCRIBE</h2>
                        <p style="font-size: 1.5em;">Confidence: {confidence:.1f}%</p>
                        <p>High potential for term deposit subscription!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no">
                        <h2>❌ UNLIKELY TO SUBSCRIBE</h2>
                        <p style="font-size: 1.5em;">Confidence: {confidence:.1f}%</p>
                        <p>May need additional engagement strategies.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown("#### Probability")
                prob_df = pd.DataFrame({
                    'Outcome': ['No', 'Yes'],
                    'Probability': [proba[0]*100, proba[1]*100]
                })
                st.bar_chart(prob_df.set_index('Outcome'))
            
            # Insights
            st.markdown("### 💡 Key Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                if poutcome == 'success':
                    st.success("✅ Previous success - high conversion potential")
                if emp_var_rate < 0:
                    st.info("📈 Negative emp.var.rate often correlates with higher subscription")
                if contact == 'cellular':
                    st.info("📱 Cellular contact tends to be more effective")
            
            with insights_col2:
                if campaign > 5:
                    st.warning("⚠️ Many contacts this campaign - may indicate resistance")
                if cons_conf_idx < -40:
                    st.warning("📉 Low consumer confidence may affect decisions")
        
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏦 <strong>Bank Marketing Prediction System</strong></p>
        <p>Gradient Boosting Model | UCI Bank Marketing Dataset</p>
        <p>CAI2C08 - Machine Learning for Developers | Temasek Polytechnic</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
