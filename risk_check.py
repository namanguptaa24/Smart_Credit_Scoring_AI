import joblib
import pandas as pd

model = joblib.load("credit_model.pkl")

def predict_risk(applicant):
    df = pd.DataFrame([applicant])
    prob = model.predict_proba(df)[0][1]

    if prob >= 0.75:
        risk = "High Risk — Review / Reject"
    elif prob >= 0.45:
        risk = "Moderate Risk — Need Manual Check"
    else:
        risk = "Low Risk — Eligible for Loan"

    return prob, risk


applicant_A = {
    "has_smartphone": 1,
    "mobile_recharge_regular": 5,
    "bill_payment_timely": 1,
    "digital_payments_freq": 8,
    "late_payments_count": 0,
    "income_level": 4,
    "employment_stability": 3
}

prob, risk = predict_risk(applicant_A)

print("Applicant Risk Probability:", round(prob, 3))
print("Decision:", risk)
