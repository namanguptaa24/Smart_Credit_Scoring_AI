import streamlit as st
import pandas as pd
import joblib
import shap

model = joblib.load("real_credit_model.pkl")

st.title("AI Credit Scoring — Financial Inclusion")
st.write("Behaviour-Driven Credit Risk Evaluation")

# Input fields
has_smartphone = st.selectbox("Has Smartphone", [0,1])
bill_payment_timely = st.selectbox("Pays Bills on Time", [0,1])
digital_payments_freq = st.slider("Digital Payment Frequency", 0,10,3)
late_payments_count = st.slider("Late Payments Count", 0,6,1)
income_level = st.slider("Income Level", 1,5,3)
employment_stability = st.slider("Employment Stability", 1,5,3)
spending_stability = st.slider("Spending Stability Indicator", 0,50000,10000)

input_data = {
    "has_smartphone": has_smartphone,
    "bill_payment_timely": bill_payment_timely,
    "digital_payments_freq": digital_payments_freq,
    "late_payments_count": late_payments_count,
    "income_level": income_level,
    "employment_stability": employment_stability,
    "spending_stability": spending_stability
}

df = pd.DataFrame([input_data])

if st.button("Predict Credit Risk"):
    prob = model.predict_proba(df)[0][1]

    if prob >= 0.75:
        decision = "High Risk — Review / Reject"
        color = "red"
    elif prob >= 0.45:
        decision = "Moderate Risk — Manual Review"
        color = "orange"
    else:
        decision = "Low Risk — Eligible for Loan"
        color = "green"

    st.subheader(f"Risk Probability: {round(prob,3)}")
    st.markdown(f"### <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)

    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    st.subheader("Why this decision was made (Feature Impact)")
    st.write("Positive values = higher risk | Negative values = lower risk")

    st.bar_chart(pd.Series(shap_values[1][0], index=df.columns))
