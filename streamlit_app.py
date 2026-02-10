"""
Bank Marketing Term Deposit Prediction App
Author: Davian | CAI2C08 - Machine Learning for Developers | Temasek Polytechnic
Dataset: UCI Bank Marketing Dataset (with Economic Indicators, Duration Removed)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .header-box {
        background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .header-box h1 {
        color: white !important;
        font-size: 28px;
        margin: 0;
    }
    
    .header-box p {
        color: #a0c4e8 !important;
        font-size: 13px;
        margin: 5px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load('bank_marketing_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'bank_marketing_model.pkl' not found!")
        return None


def create_features(inputs):
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


def create_donut_chart(probability):
    """Create a sleek donut chart for probability visualization."""
    
    color = "#10b981" if probability >= 0.5 else "#ef4444"
    
    fig = go.Figure(data=[go.Pie(
        values=[probability * 100, (1 - probability) * 100],
        hole=0.7,
        marker_colors=[color, "#e5e7eb"],
        textinfo='none',
        hoverinfo='skip',
        direction='clockwise',
        sort=False
    )])
    
    fig.add_annotation(
        text=f"<b>{probability*100:.0f}%</b>",
        x=0.5, y=0.5,
        font=dict(size=32, color=color, family="Arial"),
        showarrow=False
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        width=200
    )
    
    return fig


def get_risk_factors(inputs, proba):
    """Analyze inputs and return risk factors."""
    factors = []
    
    if inputs['poutcome'] == 'success':
        factors.append(("✅", "Previous campaign was successful", "positive"))
    if inputs['contact'] == 'cellular':
        factors.append(("✅", "Cellular contact has higher success rate", "positive"))
    if inputs['emp.var.rate'] < 0:
        factors.append(("✅", "Favorable employment conditions", "positive"))
    if inputs['education'] in ['university.degree', 'professional.course']:
        factors.append(("✅", "Higher education level", "positive"))
    if inputs['campaign'] > 5:
        factors.append(("⚠️", f"High contact attempts ({inputs['campaign']}x)", "warning"))
    if inputs['default'] == 'yes':
        factors.append(("⚠️", "Has credit default history", "warning"))
    if inputs['pdays'] == 999:
        factors.append(("ℹ️", "No previous campaign contact", "neutral"))
    if inputs['housing'] == 'yes' and inputs['loan'] == 'yes':
        factors.append(("⚠️", "Has multiple loan obligations", "warning"))
    
    return factors[:5]


def generate_report(inputs, prediction, proba, factors):
    """Generate a text report for download."""
    report = f"""
================================================================================
                    BANK MARKETING PREDICTION REPORT
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION: {'LIKELY TO SUBSCRIBE' if prediction == 'yes' else 'UNLIKELY TO SUBSCRIBE'}
Confidence: {max(proba)*100:.1f}%

PROBABILITIES
- Subscribe: {proba[1]*100:.1f}%
- Not Subscribe: {proba[0]*100:.1f}%

CLIENT PROFILE
- Age: {inputs['age']} | Job: {inputs['job']} | Education: {inputs['education']}
- Marital: {inputs['marital']} | Default: {inputs['default']}
- Housing Loan: {inputs['housing']} | Personal Loan: {inputs['loan']}

CAMPAIGN INFO
- Contact: {inputs['contact']} | Month: {inputs['month']} | Day: {inputs['day_of_week']}
- Contacts: {inputs['campaign']} | Previous: {inputs['previous']} | Outcome: {inputs['poutcome']}

ECONOMIC INDICATORS
- Emp.Var.Rate: {inputs['emp.var.rate']} | CPI: {inputs['cons.price.idx']}
- CCI: {inputs['cons.conf.idx']} | Euribor: {inputs['euribor3m']} | Employed: {inputs['nr.employed']}k

KEY FACTORS
"""
    for icon, text, _ in factors:
        report += f"  {icon} {text}\n"
    
    report += """
================================================================================
CAI2C08 - Machine Learning for Developers | Temasek Polytechnic
================================================================================
"""
    return report


def main():
    # Header
    st.markdown("""
    <div class="header-box">
        <h1>🏦 Bank Marketing Term Deposit Predictor</h1>
        <p>UCI Dataset | Economic Indicators Included | Duration Removed (No Data Leakage)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()
    
    metrics = model_package.get('metrics', {})
    threshold = model_package.get('threshold', 0.5)
    
    # Model Metrics
    with st.expander("📊 View Model Performance Metrics", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}%")
        col2.metric("F1 Score", f"{metrics.get('f1_score', 'N/A')}%")
        col3.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
        col4.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
    
    # Client Information
    st.subheader("📝 Client Information")
    
    # Personal Details
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
    
    # Financial Status
    st.markdown("**💳 Financial Status**")
    col1, col2, col3 = st.columns(3)
    with col1:
        default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
    with col2:
        housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
    with col3:
        loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
    
    # Campaign Info
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
        campaign = st.slider("Contacts (This Campaign)", 1, 50, 2)
    
    # Previous Campaign
    st.markdown("**📜 Previous Campaign**")
    col1, col2, col3 = st.columns(3)
    with col1:
        pdays = st.selectbox("Days Since Last Contact", 
                              [999, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90],
                              help="999 = never contacted before")
    with col2:
        previous = st.slider("Previous Contacts", 0, 10, 0)
    with col3:
        poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
    
    # Economic Indicators
    st.markdown("**📈 Economic Indicators**")
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
    
    st.divider()
    
    # Collect inputs
    inputs = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    
    # PREDICT BUTTON
    if st.button("🔮 PREDICT SUBSCRIPTION LIKELIHOOD", type="primary", use_container_width=True):
        
        try:
            input_df = create_features(inputs)
            input_df = input_df[model_package['feature_columns']]
            
            X_processed = model_package['preprocessor'].transform(input_df)
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            prediction = 1 if proba[1] >= threshold else 0
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            
            factors = get_risk_factors(inputs, proba)
            
            # ============ RESULT DISPLAY ============
            st.markdown("---")
            
            if pred_label == 'yes':
                st.balloons()
                
                # Result container
                result_container = st.container()
                with result_container:
                    st.success("## 🎯 HIGH POTENTIAL CLIENT")
                    st.markdown("##### This client is **likely to subscribe** to a term deposit")
                    
                    # Probability display
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric(
                            label="Subscribe Probability",
                            value=f"{proba[1]*100:.1f}%",
                            delta="Positive"
                        )
                    with col2:
                        # Donut chart
                        fig = create_donut_chart(proba[1])
                        st.plotly_chart(fig, use_container_width=True)
                    with col3:
                        st.metric(
                            label="Not Subscribe Probability", 
                            value=f"{proba[0]*100:.1f}%"
                        )
            else:
                # Result container
                result_container = st.container()
                with result_container:
                    st.error("## 📊 LOW POTENTIAL CLIENT")
                    st.markdown("##### This client is **unlikely to subscribe** to a term deposit")
                    
                    # Probability display
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric(
                            label="Subscribe Probability",
                            value=f"{proba[1]*100:.1f}%"
                        )
                    with col2:
                        # Donut chart
                        fig = create_donut_chart(proba[1])
                        st.plotly_chart(fig, use_container_width=True)
                    with col3:
                        st.metric(
                            label="Not Subscribe Probability",
                            value=f"{proba[0]*100:.1f}%",
                            delta="Higher",
                            delta_color="inverse"
                        )
            
            # Key Factors
            st.markdown("### 💡 Key Factors")
            
            for icon, text, factor_type in factors:
                if factor_type == "positive":
                    st.success(f"{icon} {text}")
                elif factor_type == "warning":
                    st.warning(f"{icon} {text}")
                else:
                    st.info(f"{icon} {text}")
            
            # Download Report
            st.markdown("---")
            report = generate_report(inputs, pred_label, proba, factors)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📄 Download Prediction Report",
                    data=report,
                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    
    # Footer
    st.divider()
    st.caption("Bank Marketing Prediction | CAI2C08 - ML for Developers | Temasek Polytechnic | Davian")


if __name__ == "__main__":
    main()
