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
    
    .profile-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #2b6cb0;
    }
    
    .insight-card {
        background: #f0f9ff;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #bae6fd;
        margin: 5px 0;
    }
    
    .metric-highlight {
        font-size: 32px;
        font-weight: 700;
        color: #2b6cb0;
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


def create_gauge_chart(probability, title="Subscription Likelihood"):
    """Create a professional gauge chart for prediction probability."""
    
    # Determine color based on probability
    if probability >= 0.5:
        bar_color = "#10b981"  # Green
    elif probability >= 0.3:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': "%", 'font': {'size': 40, 'color': '#1a365d'}},
        title={'text': title, 'font': {'size': 16, 'color': '#64748b'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b",
                     'tickvals': [0, 25, 50, 75, 100],
                     'ticktext': ['0%', '25%', '50%', '75%', '100%']},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#fee2e2'},
                {'range': [30, 50], 'color': '#fef3c7'},
                {'range': [50, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#1a365d", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Arial"}
    )
    
    return fig


def create_comparison_chart(prob_yes, prob_no):
    """Create a horizontal bar chart comparing probabilities."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['Prediction'],
        x=[prob_yes * 100],
        name='Subscribe',
        orientation='h',
        marker=dict(color='#10b981', line=dict(color='#059669', width=1)),
        text=[f'{prob_yes*100:.1f}%'],
        textposition='inside',
        textfont=dict(color='white', size=14)
    ))
    
    fig.add_trace(go.Bar(
        y=['Prediction'],
        x=[prob_no * 100],
        name='Not Subscribe',
        orientation='h',
        marker=dict(color='#ef4444', line=dict(color='#dc2626', width=1)),
        text=[f'{prob_no*100:.1f}%'],
        textposition='inside',
        textfont=dict(color='white', size=14)
    ))
    
    fig.update_layout(
        barmode='stack',
        height=100,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig


def get_risk_factors(inputs, proba):
    """Analyze inputs and return risk factors."""
    factors = []
    
    # Positive factors
    if inputs['poutcome'] == 'success':
        factors.append(("✅", "Previous campaign was successful", "positive"))
    if inputs['contact'] == 'cellular':
        factors.append(("✅", "Contacted via cellular (higher success rate)", "positive"))
    if inputs['emp.var.rate'] < 0:
        factors.append(("✅", "Favorable employment conditions", "positive"))
    if inputs['education'] in ['university.degree', 'professional.course']:
        factors.append(("✅", "Higher education level", "positive"))
    if inputs['previous'] > 0:
        factors.append(("✅", "Previously engaged with bank", "positive"))
    
    # Negative factors
    if inputs['campaign'] > 5:
        factors.append(("⚠️", f"High contact attempts ({inputs['campaign']}x) may indicate resistance", "warning"))
    if inputs['default'] == 'yes':
        factors.append(("⚠️", "Has credit default history", "warning"))
    if inputs['pdays'] == 999:
        factors.append(("ℹ️", "No previous campaign contact", "neutral"))
    if inputs['cons.conf.idx'] < -45:
        factors.append(("⚠️", "Low consumer confidence period", "warning"))
    if inputs['housing'] == 'yes' and inputs['loan'] == 'yes':
        factors.append(("⚠️", "Has both housing and personal loans", "warning"))
    
    return factors


def generate_report(inputs, prediction, proba, factors):
    """Generate a text report for download."""
    report = f"""
================================================================================
                    BANK MARKETING PREDICTION REPORT
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION SUMMARY
------------------
Result: {'LIKELY TO SUBSCRIBE' if prediction == 'yes' else 'UNLIKELY TO SUBSCRIBE'}
Subscription Probability: {proba[1]*100:.1f}%
Non-Subscription Probability: {proba[0]*100:.1f}%
Confidence: {max(proba)*100:.1f}%

CLIENT PROFILE
--------------
Age: {inputs['age']}
Job: {inputs['job']}
Marital Status: {inputs['marital']}
Education: {inputs['education']}
Credit Default: {inputs['default']}
Housing Loan: {inputs['housing']}
Personal Loan: {inputs['loan']}

CAMPAIGN INFORMATION
--------------------
Contact Type: {inputs['contact']}
Month: {inputs['month']}
Day of Week: {inputs['day_of_week']}
Campaign Contacts: {inputs['campaign']}
Days Since Last Contact: {inputs['pdays']}
Previous Contacts: {inputs['previous']}
Previous Outcome: {inputs['poutcome']}

ECONOMIC INDICATORS
-------------------
Employment Variation Rate: {inputs['emp.var.rate']}
Consumer Price Index: {inputs['cons.price.idx']}
Consumer Confidence Index: {inputs['cons.conf.idx']}
Euribor 3M Rate: {inputs['euribor3m']}
Number Employed (thousands): {inputs['nr.employed']}

KEY FACTORS
-----------
"""
    for icon, text, _ in factors:
        report += f"{icon} {text}\n"
    
    report += """
================================================================================
                         END OF REPORT
================================================================================
Note: This prediction is based on historical data patterns and should be used
as one of many factors in decision-making. Model accuracy may vary.

Dataset: UCI Bank Marketing Dataset
Model: Gradient Boosting with SMOTE
Institution: Temasek Polytechnic - CAI2C08
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
    
    # Model Metrics in expander
    with st.expander("📊 View Model Performance Metrics", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A')}%")
        col2.metric("F1 Score", f"{metrics.get('f1_score', 'N/A')}%")
        col3.metric("Precision", f"{metrics.get('precision', 'N/A')}%")
        col4.metric("Recall", f"{metrics.get('recall', 'N/A')}%")
        col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
        st.caption(f"Decision Threshold: {threshold}")
    
    # Two-column layout
    col_input, col_result = st.columns([3, 2])
    
    with col_input:
        st.subheader("📝 Client Information")
        
        # Personal Details
        st.markdown("**👤 Personal Details**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.slider("Age", 18, 95, 35)
        with c2:
            job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                        'management', 'retired', 'self-employed', 'services', 
                                        'student', 'technician', 'unemployed', 'unknown'])
        with c3:
            marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
        with c4:
            education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                    'illiterate', 'professional.course', 
                                                    'university.degree', 'unknown'])
        
        # Financial Status
        st.markdown("**💳 Financial Status**")
        c1, c2, c3 = st.columns(3)
        with c1:
            default = st.selectbox("Credit Default?", ['no', 'yes', 'unknown'])
        with c2:
            housing = st.selectbox("Housing Loan?", ['no', 'yes', 'unknown'])
        with c3:
            loan = st.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])
        
        # Campaign Info
        st.markdown("**📞 Campaign Information**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        with c2:
            month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
        with c3:
            day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
        with c4:
            campaign = st.slider("Contacts (This Campaign)", 1, 50, 2)
        
        # Previous Campaign
        st.markdown("**📜 Previous Campaign**")
        c1, c2, c3 = st.columns(3)
        with c1:
            pdays = st.selectbox("Days Since Last Contact", 
                                  [999, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90],
                                  help="999 = never contacted before")
        with c2:
            previous = st.slider("Previous Contacts", 0, 10, 0)
        with c3:
            poutcome = st.selectbox("Previous Outcome", ['nonexistent', 'failure', 'success'])
        
        # Economic Indicators
        st.markdown("**📈 Economic Indicators**")
        c1, c2, c3 = st.columns(3)
        with c1:
            emp_var_rate = st.slider("Emp. Var. Rate", -3.5, 1.5, -0.1, 0.1)
            cons_price_idx = st.slider("Consumer Price Idx", 92.0, 95.0, 93.5, 0.1)
        with c2:
            cons_conf_idx = st.slider("Consumer Conf. Idx", -51.0, -26.0, -40.0, 0.5)
            euribor3m = st.slider("Euribor 3M Rate", 0.5, 5.1, 2.5, 0.1)
        with c3:
            nr_employed = st.selectbox("Nr. Employed (k)", 
                                        [4963.6, 5008.7, 5017.5, 5076.2, 5099.1, 
                                         5176.3, 5191.0, 5195.8, 5228.1], index=4)
    
    # Collect inputs
    inputs = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
        'month': month, 'day_of_week': day_of_week, 'campaign': campaign,
        'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    
    with col_result:
        st.subheader("🎯 Prediction Result")
        
        # Always show prediction (live update)
        try:
            input_df = create_features(inputs)
            input_df = input_df[model_package['feature_columns']]
            
            X_processed = model_package['preprocessor'].transform(input_df)
            proba = model_package['model'].predict_proba(X_processed)[0]
            
            prediction = 1 if proba[1] >= threshold else 0
            pred_label = model_package['label_encoder'].inverse_transform([prediction])[0]
            
            # Gauge Chart
            fig = create_gauge_chart(proba[1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Verdict
            if pred_label == 'yes':
                st.success(f"### ✅ LIKELY TO SUBSCRIBE")
            else:
                st.error(f"### ❌ UNLIKELY TO SUBSCRIBE")
            
            # Probability Bar
            st.markdown("**Probability Breakdown**")
            fig2 = create_comparison_chart(proba[1], proba[0])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Risk Factors
            st.markdown("**Key Factors**")
            factors = get_risk_factors(inputs, proba)
            for icon, text, factor_type in factors[:5]:  # Show top 5
                if factor_type == "positive":
                    st.markdown(f'<div class="insight-card" style="border-left: 3px solid #10b981;">{icon} {text}</div>', unsafe_allow_html=True)
                elif factor_type == "warning":
                    st.markdown(f'<div class="insight-card" style="border-left: 3px solid #f59e0b;">{icon} {text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="insight-card" style="border-left: 3px solid #64748b;">{icon} {text}</div>', unsafe_allow_html=True)
            
            # Download Report
            st.markdown("---")
            report = generate_report(inputs, pred_label, proba, factors)
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
