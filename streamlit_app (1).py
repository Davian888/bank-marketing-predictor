"""
Bank Marketing Term Deposit Prediction
Streamlit Web Application

CAI2C08 - Machine Learning for Developers Project
Temasek Polytechnic

This application predicts whether a customer will subscribe to a term deposit
based on their demographic information and campaign interaction history.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1a5276, #2980b9);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #2980b9;
    }
    
    /* Result boxes */
    .result-yes {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46,204,113,0.4);
    }
    
    .result-no {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(231,76,60,0.4);
    }
    
    .confidence-box {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Form styling */
    .stSelectbox, .stNumberInput, .stSlider {
        background: white;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2980b9, #1a5276);
        color: white;
        padding: 0.8rem 2rem;
        font-size: 1.2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(41,128,185,0.4);
    }
    
    /* Section headers */
    .section-header {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #2980b9;
        margin: 1rem 0;
        font-weight: bold;
        color: #1a5276;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model package"""
    try:
        model_package = joblib.load('bank_marketing_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'bank_marketing_model.pkl' is in the same directory.")
        return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def categorize_age(age):
    """Categorize age into groups"""
    if age < 25:
        return 'young'
    elif age < 35:
        return 'young_adult'
    elif age < 50:
        return 'middle_aged'
    elif age < 65:
        return 'mature'
    else:
        return 'senior'

def create_features(input_data):
    """Create engineered features from input data"""
    df = pd.DataFrame([input_data])
    
    # Feature engineering (must match training)
    df['age_group'] = df['age'].apply(categorize_age)
    df['contact_intensity'] = df['campaign'] + df['previous']
    df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
    df['economic_score'] = (
        (df['cons_conf_idx'] - (-50.8)) / ((-26.9) - (-50.8)) -
        (df['euribor3m'] - 0.634) / (5.045 - 0.634)
    )
    df['has_any_loan'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
    
    return df

def get_recommendation(prediction, probability):
    """Generate recommendation based on prediction"""
    if prediction == 1:
        if probability > 0.7:
            return "HIGH PRIORITY: This customer shows strong indicators of interest. Recommend immediate follow-up with personalized offer."
        else:
            return "POTENTIAL SUBSCRIBER: Customer may be interested. Recommend scheduling a follow-up call within 1-2 days."
    else:
        if probability > 0.3:
            return "MARGINAL CASE: Customer is on the fence. Consider offering a special promotion or limited-time deal."
        else:
            return "LOW PRIORITY: Customer unlikely to subscribe. May revisit in 3-6 months or focus resources elsewhere."

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bank Marketing Predictor</h1>
        <p>Predict Customer Subscription to Term Deposits</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            This tool is for demonstration purposes only. Not for actual financial decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # Sidebar - About
    with st.sidebar:
        st.markdown("## About This Tool")
        st.markdown("""
        This application uses machine learning to predict whether a customer 
        will subscribe to a term deposit based on their profile and campaign data.
        """)
        
        st.markdown("---")
        st.markdown("### Model Performance")
        
        metrics = model_package.get('metrics', {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0.81)*100:.1f}%")
            st.metric("Precision", f"{metrics.get('precision', 0.25)*100:.1f}%")
        with col2:
            st.metric("F1 Score", f"{metrics.get('f1_score', 0.17):.3f}")
            st.metric("Recall", f"{metrics.get('recall', 0.13)*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### Key Predictors")
        st.markdown("""
        - Previous campaign outcome
        - Contact type (cellular vs telephone)
        - Job type
        - Economic indicators
        - Contact history
        """)
        
        st.markdown("---")
        st.markdown("### Resources")
        st.markdown("[Dataset Source (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)")
    
    # Main content
    st.markdown('<div class="section-header">Customer Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Demographics")
        
        age = st.number_input("Age", min_value=18, max_value=100, value=35, 
                              help="Customer's age in years")
        
        job = st.selectbox("Job Type", 
                           options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                   'management', 'retired', 'self-employed', 'services', 
                                   'student', 'technician', 'unemployed', 'unknown'],
                           help="Customer's type of job")
        
        marital = st.selectbox("Marital Status",
                               options=['married', 'single', 'divorced', 'unknown'],
                               help="Customer's marital status")
        
        education = st.selectbox("Education Level",
                                 options=['university.degree', 'high.school', 'professional.course',
                                         'basic.9y', 'basic.6y', 'basic.4y', 'illiterate', 'unknown'],
                                 help="Customer's highest education level")
    
    with col2:
        st.markdown("#### Financial Status")
        
        default = st.selectbox("Has Credit Default?",
                               options=['no', 'yes', 'unknown'],
                               help="Does customer have credit in default?")
        
        housing = st.selectbox("Has Housing Loan?",
                               options=['no', 'yes', 'unknown'],
                               help="Does customer have a housing loan?")
        
        loan = st.selectbox("Has Personal Loan?",
                            options=['no', 'yes', 'unknown'],
                            help="Does customer have a personal loan?")
    
    st.markdown('<div class="section-header">Campaign Information</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Contact Details")
        
        contact = st.selectbox("Contact Type",
                               options=['cellular', 'telephone'],
                               help="Type of contact communication")
        
        month = st.selectbox("Last Contact Month",
                             options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                     'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                             index=4,
                             help="Month of last contact")
        
        day_of_week = st.selectbox("Day of Week",
                                   options=['mon', 'tue', 'wed', 'thu', 'fri'],
                                   help="Last contact day of the week")
        
        campaign = st.slider("Contacts This Campaign", min_value=1, max_value=35, value=2,
                             help="Number of contacts during this campaign")
    
    with col4:
        st.markdown("#### Previous Campaign")
        
        poutcome = st.selectbox("Previous Campaign Outcome",
                                options=['nonexistent', 'failure', 'success'],
                                help="Outcome of the previous marketing campaign")
        
        # If previous outcome exists, ask for details
        if poutcome != 'nonexistent':
            pdays = st.slider("Days Since Last Contact", min_value=1, max_value=30, value=10,
                              help="Days since customer was last contacted from previous campaign")
            previous = st.slider("Previous Contacts", min_value=1, max_value=7, value=1,
                                 help="Number of contacts before this campaign")
        else:
            pdays = 999  # Never contacted
            previous = 0
    
    # Economic indicators (using collapsible section)
    with st.expander("Economic Indicators (Advanced)"):
        st.info("These are economic context indicators. Default values represent average conditions.")
        
        col5, col6 = st.columns(2)
        
        with col5:
            emp_var_rate = st.slider("Employment Variation Rate", 
                                     min_value=-3.4, max_value=1.4, value=-0.1, step=0.1,
                                     help="Quarterly employment variation rate")
            
            cons_price_idx = st.slider("Consumer Price Index", 
                                       min_value=92.2, max_value=94.8, value=93.5, step=0.1,
                                       help="Monthly consumer price index")
            
            cons_conf_idx = st.slider("Consumer Confidence Index", 
                                      min_value=-50.8, max_value=-26.9, value=-40.0, step=0.1,
                                      help="Monthly consumer confidence index")
        
        with col6:
            euribor3m = st.slider("Euribor 3 Month Rate", 
                                  min_value=0.634, max_value=5.045, value=3.0, step=0.1,
                                  help="Daily Euribor 3 month rate")
            
            nr_employed = st.selectbox("Number of Employees (quarterly)",
                                       options=[4963.6, 5008.7, 5076.2, 5099.1, 5176.3, 5191.0, 5195.8, 5228.1],
                                       index=4,
                                       help="Quarterly indicator - number of employees")
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Subscription Likelihood", use_container_width=True):
        
        # Prepare input data
        input_data = {
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
            'emp_var_rate': emp_var_rate,
            'cons_price_idx': cons_price_idx,
            'cons_conf_idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr_employed': nr_employed
        }
        
        # Create features
        df_input = create_features(input_data)
        
        # Get feature columns in correct order
        feature_cols = model_package['feature_columns']
        df_input = df_input[feature_cols]
        
        # Preprocess
        preprocessor = model_package['preprocessor']
        X_processed = preprocessor.transform(df_input)
        
        # Predict
        model = model_package['model']
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-yes">
                    <h2>✅ LIKELY TO SUBSCRIBE</h2>
                    <p style="font-size: 1.2rem;">This customer shows indicators of potential interest in the term deposit.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-no">
                    <h2>❌ UNLIKELY TO SUBSCRIBE</h2>
                    <p style="font-size: 1.2rem;">This customer may not be interested in the term deposit at this time.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_result2:
            st.markdown(f"""
            <div class="confidence-box">
                <h3>Confidence Score</h3>
                <p style="font-size: 2.5rem; margin: 0.5rem 0;">{max(probability)*100:.1f}%</p>
                <p style="font-size: 0.9rem;">Probability of {'subscribing' if prediction == 1 else 'not subscribing'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### Probability Breakdown")
        
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            st.metric("Not Subscribe Probability", f"{probability[0]*100:.1f}%")
            st.progress(probability[0])
        
        with col_prob2:
            st.metric("Subscribe Probability", f"{probability[1]*100:.1f}%")
            st.progress(probability[1])
        
        # Recommendation
        st.markdown("### Recommendation")
        recommendation = get_recommendation(prediction, probability[1])
        st.info(recommendation)
        
        # Customer Summary
        st.markdown("### Customer Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown(f"""
            **Demographics**
            - Age: {age} ({categorize_age(age)})
            - Job: {job}
            - Marital: {marital}
            - Education: {education}
            """)
        
        with summary_col2:
            st.markdown(f"""
            **Financial Status**
            - Default: {default}
            - Housing Loan: {housing}
            - Personal Loan: {loan}
            """)
        
        with summary_col3:
            st.markdown(f"""
            **Campaign Info**
            - Contact: {contact}
            - Previous Outcome: {poutcome}
            - Total Contacts: {campaign + previous}
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🏦 Bank Marketing Term Deposit Predictor | CAI2C08 Machine Learning Project</p>
        <p>Temasek Polytechnic | 2025</p>
        <p style="font-size: 0.8rem; color: #95a5a6;">
            Disclaimer: This is a student project for educational purposes only. 
            Not intended for actual financial or business decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    main()
