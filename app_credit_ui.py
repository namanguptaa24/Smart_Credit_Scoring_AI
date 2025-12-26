import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_model.pkl")

st.title("Smart Credit Scoring — Financial Inclusion AI")
st.write("Behaviour-driven credit risk evaluation using alternative data")

# --- INPUT FORM ---
has_smartphone = st.selectbox("Has Smartphone", [0, 1])
mobile_recharge_regular = st.slider("Mobile Recharge Regularity", 1, 5, 3)
bill_payment_timely = st.selectbox("Pays Bills on Time", [0, 1])
digital_payments_freq = st.slider("Digital Payment Frequency", 1, 10, 5)
late_payments_count = st.slider("Late Payments Count", 0, 5, 1)
income_level = st.slider("Income Level", 1, 5, 3)
employment_stability = st.slider("Employment Stability", 1, 4, 2)

input_data = {
    "has_smartphone": has_smartphone,
    "mobile_recharge_regular": mobile_recharge_regular,
    "bill_payment_timely": bill_payment_timely,
    "digital_payments_freq": digital_payments_freq,
    "late_payments_count": late_payments_count,
    "income_level": income_level,
    "employment_stability": employment_stability
}

if st.button("Predict Credit Risk"):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]

    if prob >= 0.75:
        result = "High Risk — Review / Reject"
        color = "red"
    elif prob >= 0.45:
        result = "Moderate Risk — Manual Review Suggested"
        color = "orange"
    else:
        result = "Low Risk — Eligible for Loan"
        color = "green"

    st.subheader(f"Risk Probability: {round(prob,3)}")
    st.markdown(f"### <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
