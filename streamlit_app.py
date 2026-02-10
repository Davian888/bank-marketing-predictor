"""
Bank Marketing Term Deposit Prediction App
============================================
A machine learning application to predict customer subscription likelihood
for bank term deposits based on demographic and campaign data.

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

# Custom CSS for professional styling - ORIGINAL STYLE with label fix
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Card styling - ORIGINAL purple/pink gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 10px 0;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 15px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5em;
        margin-bottom: 5px;
    }
    
    /* LABEL VISIBILITY FIX */
    label, .stSelectbox label, .stSlider label, .stNumberInput label,
    .stSelectbox > label, .stSlider > label, .stNumberInput > label,
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] {
        color: #1e3c72 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }
    
    /* Result boxes */
    .result-yes {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
    }
    
    .result-no {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(235, 51, 73, 0.4);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(17, 153, 142, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #fff9c4, #fff59d);
        border-left: 5px solid #fbc02d;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #1e3c72;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and preprocessor."""
    try:
        model_package = joblib.load('bank_marketing_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'bank_marketing_model.pkl' is in the same directory.")
        return None


def create_feature_dataframe(inputs):
    """Create a DataFrame from user inputs with feature engineering."""
    df = pd.DataFrame([inputs])
    
    # Feature engineering (must match training - NO DURATION)
    df['was_contacted_before'] = (df['previous'] > 0).astype(int)
    df['has_loan_burden'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
    df['high_balance'] = (df['balance'] > 1300).astype(int)  # Using approximate median
    
    # Age groups
    def age_group(age):
        if age < 25: return 'young'
        elif age < 35: return 'young_adult'
        elif age < 50: return 'middle_age'
        elif age < 65: return 'senior'
        else: return 'elderly'
    
    df['age_group'] = df['age'].apply(age_group)
    
    return df


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Bank Marketing Predictor</h1>
        <p style="font-size: 1.2em; opacity: 0.9;">Predict Term Deposit Subscription Likelihood</p>
        <p style="font-size: 0.9em; opacity: 0.7;">Machine Learning Classification Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # GET METRICS FROM PKL FILE - NO MORE HARDCODING!
    metrics = model_package.get('metrics', {})
    accuracy = metrics.get('accuracy', 'N/A')
    f1_score_val = metrics.get('f1_score', 'N/A')
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 About This Tool")
        st.info("""
        This app predicts whether a bank client will subscribe to a term deposit 
        based on their demographic information and campaign contact history.
        """)
        
        st.markdown("### 🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            # DYNAMIC - reads from pkl!
            st.metric("Accuracy", f"{accuracy}%")
        with col2:
            # DYNAMIC - reads from pkl!
            st.metric("F1 Score", f"{f1_score_val}%")
        
        st.markdown("### 📈 Key Predictors")
        st.markdown("""
        - ✅ Previous Campaign Success
        - 💰 Account Balance
        - 👤 Age Group
        - 📱 Contact Method
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Dataset Info")
        st.markdown("""
        **Source:** UCI Machine Learning Repository  
        **Samples:** 4,521  
        **Features:** 16 + 4 engineered
        """)
        
        st.markdown("---")
        st.caption("CAI2C08 - Machine Learning for Developers")
        st.caption("Temasek Polytechnic © 2025")
    
    # Main content area
    st.markdown("### 📝 Enter Client Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Personal Details")
        
        age = st.slider("Age", min_value=18, max_value=95, value=35, help="Client's age in years")
        
        job = st.selectbox("Job Type", options=[
            'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician',
            'unemployed', 'unknown'
        ], help="Type of occupation")
        
        marital = st.selectbox("Marital Status", options=['single', 'married', 'divorced'],
                               help="Marital status of the client")
        
        education = st.selectbox("Education Level", options=['primary', 'secondary', 'tertiary', 'unknown'],
                                 help="Highest education level achieved")
        
        st.markdown("#### 💳 Financial Status")
        
        balance = st.number_input("Average Yearly Balance (€)", min_value=-10000, max_value=100000, 
                                  value=1500, step=100, help="Average yearly balance in euros")
        
        default = st.selectbox("Has Credit in Default?", options=['no', 'yes'],
                               help="Does the client have credit in default?")
        
        housing = st.selectbox("Has Housing Loan?", options=['no', 'yes'],
                               help="Does the client have a housing loan?")
        
        loan = st.selectbox("Has Personal Loan?", options=['no', 'yes'],
                            help="Does the client have a personal loan?")
    
    with col2:
        st.markdown("#### 📞 Campaign Information")
        
        contact = st.selectbox("Contact Method", options=['cellular', 'telephone', 'unknown'],
                               help="Method of communication")
        
        month = st.selectbox("Last Contact Month", options=[
            'jan', 'feb', 'mar', 'apr', 'may', 'jun',
            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ], index=4, help="Month of last contact")
        
        day = st.slider("Last Contact Day", min_value=1, max_value=31, value=15,
                        help="Day of the month of last contact")
        
        campaign = st.number_input("Contacts This Campaign", min_value=1, max_value=50,
                                   value=2, help="Number of contacts during this campaign")
        
        st.markdown("#### 📜 Previous Campaign History")
        
        pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=999,
                                value=-1, help="Days since last contact (-1 = never contacted)")
        
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50,
                                   value=0, help="Number of contacts before this campaign")
        
        poutcome = st.selectbox("Previous Campaign Outcome", 
                                options=['unknown', 'failure', 'other', 'success'],
                                help="Outcome of previous marketing campaign")
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Subscription Likelihood", use_container_width=True):
        # Collect inputs (NO DURATION!)
        inputs = {
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }
        
        # Create feature dataframe with engineering
        input_df = create_feature_dataframe(inputs)
        
        # Ensure column order matches training
        feature_cols = model_package['feature_columns']
        input_df = input_df[feature_cols]
        
        # Preprocess and predict
        try:
            X_processed = model_package['preprocessor'].transform(input_df)
            prediction = model_package['model'].predict(X_processed)[0]
            probability = model_package['model'].predict_proba(X_processed)[0]
            
            # Decode prediction
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            confidence = probability[prediction] * 100
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Display result
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                if pred_label == 'yes':
                    st.markdown(f"""
                    <div class="result-yes">
                        <h2>✅ LIKELY TO SUBSCRIBE</h2>
                        <p style="font-size: 1.5em;">Confidence: {confidence:.1f}%</p>
                        <p>This client shows high potential for term deposit subscription!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no">
                        <h2>❌ UNLIKELY TO SUBSCRIBE</h2>
                        <p style="font-size: 1.5em;">Confidence: {confidence:.1f}%</p>
                        <p>This client may need additional engagement strategies.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("#### 📊 Probability Distribution")
                prob_df = pd.DataFrame({
                    'Outcome': ['No Subscription', 'Subscription'],
                    'Probability': [probability[0] * 100, probability[1] * 100]
                })
                st.bar_chart(prob_df.set_index('Outcome'))
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("#### 🎯 Key Insights")
                insights = []
                
                if poutcome == 'success':
                    insights.append("✅ Previous success - high conversion potential")
                elif poutcome == 'failure':
                    insights.append("⚠️ Previous failure - may need different approach")
                
                if balance > 2000:
                    insights.append("✅ Good account balance - financially stable")
                elif balance < 0:
                    insights.append("⚠️ Negative balance - may not be ideal candidate")
                
                if job in ['student', 'retired']:
                    insights.append("✅ Job type associated with higher subscription rates")
                
                for insight in insights:
                    st.markdown(insight)
            
            with rec_col2:
                st.markdown("#### 📋 Next Steps")
                if pred_label == 'yes':
                    st.success("""
                    **Recommended Actions:**
                    1. Schedule follow-up call within 48 hours
                    2. Prepare detailed term deposit benefits
                    3. Offer personalized interest rates
                    4. Provide comparison with other investment options
                    """)
                else:
                    st.warning("""
                    **Improvement Strategies:**
                    1. Build relationship with multiple contacts
                    2. Wait for better economic conditions
                    3. Target during optimal months (Mar, Oct, Dec)
                    4. Consider alternative products first
                    """)
            
            # Client Summary
            with st.expander("📋 View Client Summary"):
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.markdown("**Demographics**")
                    st.write(f"- Age: {age} years")
                    st.write(f"- Job: {job}")
                    st.write(f"- Education: {education}")
                    st.write(f"- Marital: {marital}")
                
                with summary_col2:
                    st.markdown("**Financial**")
                    st.write(f"- Balance: €{balance:,}")
                    st.write(f"- Default: {default}")
                    st.write(f"- Housing Loan: {housing}")
                    st.write(f"- Personal Loan: {loan}")
                
                with summary_col3:
                    st.markdown("**Campaign**")
                    st.write(f"- Contact: {contact}")
                    st.write(f"- Contacts: {campaign}")
                    st.write(f"- Previous: {poutcome}")
        
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")
            st.info("Please check that all inputs are valid and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏦 <strong>Bank Marketing Prediction System</strong></p>
        <p>Built with Streamlit | Machine Learning Model: Random Forest Classifier</p>
        <p>CAI2C08 - Machine Learning for Developers | Temasek Polytechnic</p>
        <p><em>⚠️ This is a predictive tool for educational purposes. 
        Actual subscription outcomes may vary based on additional factors.</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
