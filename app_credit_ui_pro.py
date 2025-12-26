import streamlit as st
import pandas as pd
import joblib
import shap
import json
from io import BytesIO

# ---------------- Load Model ----------------
model = joblib.load("real_credit_model.pkl")

st.title("AI Credit Scoring — Financial Inclusion Platform")
st.write("Behaviour-driven Credit Risk Evaluation with Explainable AI")

# ---------------- Applicant Input ----------------
st.sidebar.header("Applicant Profile Input")

applicant_id = st.sidebar.text_input("Applicant ID / Case ID", "APP-101")

has_smartphone = st.sidebar.selectbox("Has Smartphone", [0,1])
bill_payment_timely = st.sidebar.selectbox("Pays Bills on Time", [0,1])
digital_payments_freq = st.sidebar.slider("Digital Payment Frequency", 0,10,3)
late_payments_count = st.sidebar.slider("Late Payments Count", 0,6,1)
income_level = st.sidebar.slider("Income Level", 1,5,3)
employment_stability = st.sidebar.slider("Employment Stability", 1,5,3)
spending_stability = st.sidebar.slider("Spending Stability Indicator", 0,50000,10000)

input_data = {
    "Applicant_ID": applicant_id,
    "has_smartphone": has_smartphone,
    "bill_payment_timely": bill_payment_timely,
    "digital_payments_freq": digital_payments_freq,
    "late_payments_count": late_payments_count,
    "income_level": income_level,
    "employment_stability": employment_stability,
    "spending_stability": spending_stability
}

df = pd.DataFrame([input_data]).drop(columns=["Applicant_ID"])

# ---------------- Prediction ----------------
if st.button("Predict Credit Risk"):

    prob = model.predict_proba(df)[0][1]
    confidence = round(abs(prob - 0.5) * 2, 3)

    if prob >= 0.75:
        decision = "High Risk — Review / Reject"
        color = "red"
    elif prob >= 0.45:
        decision = "Moderate Risk — Manual Review"
        color = "orange"
    else:
        decision = "Low Risk — Eligible for Loan"
        color = "green"

    st.subheader("📊 Credit Risk Prediction")
    st.markdown(
        f"""
        **Applicant ID:** `{applicant_id}`  

        **Risk Probability:** `{round(prob,3)}`  
        **Model Confidence:** `{confidence}`  

        ### <span style='color:{color}'>{decision}</span>
        """,
        unsafe_allow_html=True
    )

    # ---------------- SHAP Explainability ----------------
    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    shap_series = pd.Series(shap_values.values[0], index=df.columns)

    st.subheader("🧠 Explainable AI — Feature Influence")
    st.caption("Positive = increases risk | Negative = reduces risk")
    st.bar_chart(shap_series)

    # ---------------- Download Prediction Report ----------------
    st.subheader("📥 Download Prediction Report")

    report = {
        "Applicant_ID": applicant_id,
        "Decision": decision,
        "Risk_Probability": float(prob),
        "Model_Confidence": float(confidence),
        "Feature_Inputs": input_data,
        "Feature_Impact": shap_series.to_dict()
    }

    buffer = BytesIO()
    buffer.write(json.dumps(report, indent=4).encode())
    buffer.seek(0)

    st.download_button(
        label="Download Report (JSON)",
        data=buffer,
        file_name=f"{applicant_id}_risk_report.json",
        mime="application/json"
    )

    # ---------------- Officer Review Screen ----------------
    st.subheader("🏦 Loan Officer Review Panel")

    review_choice = st.radio(
        "Officer Action",
        ["Approve", "Approve with Guarantee", "Manual Review Required", "Reject"]
    )

    comments = st.text_area("Officer Remarks (Reasoning / Observations)")

    if st.button("Save Officer Decision"):
        st.success("Officer decision saved (for prototype).")
        st.write(f"**Decision:** {review_choice}")
        st.write(f"**Remarks:** {comments}")

    # ---------------- Applicant Profile History (Mock DB) ----------------
    st.subheader("📂 Applicant Profile History")

    try:
        history_df = pd.read_csv("applicant_history.csv")
    except:
        history_df = pd.DataFrame(columns=[
            "Applicant_ID","Risk","Decision","Probability"
        ])

    new_entry = {
        "Applicant_ID": applicant_id,
        "Risk": decision,
        "Decision": review_choice if comments else "Auto Model Decision",
        "Probability": round(prob,3)
    }

    history_df = history_df.append(new_entry, ignore_index=True)
    history_df.to_csv("applicant_history.csv", index=False)

    st.dataframe(history_df.tail(5))

    # ---------------- Fairness & Bias Check (Basic Audit) ----------------
    st.subheader("⚖ Fairness & Bias Check (Prototype)")

    fairness_metrics = pd.DataFrame({
        "Feature": df.columns,
        "Risk_Correlation_Proxy": shap_series.values
    })

    st.caption("Higher absolute impact = greater influence on decision")
    st.dataframe(fairness_metrics)

    # ---------------- Model Confidence Explanation ----------------
    st.subheader("🔎 Model Confidence Explanation")

    if confidence < 0.3:
        st.warning("Low confidence — recommend manual review.")
    elif confidence < 0.6:
        st.info("Moderate confidence — acceptable but borderline.")
    else:
        st.success("High confidence — decision is reliable.")

    # ---------------- Comparison vs Previous Loans ----------------
    st.subheader("📈 Compare vs Previous Applicant Cases")

    if len(history_df) > 1:
        chart_data = history_df.tail(10)[["Applicant_ID","Probability"]]
        chart_data = chart_data.set_index("Applicant_ID")
        st.line_chart(chart_data)
    else:
        st.caption("Not enough historical applicants yet for comparison.")
