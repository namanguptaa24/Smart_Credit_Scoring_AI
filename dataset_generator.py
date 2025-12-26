import pandas as pd
import numpy as np

np.random.seed(42)

n = 300  # dataset size

data = {
    "has_smartphone": np.random.choice([0, 1], n, p=[0.2, 0.8]),
    "mobile_recharge_regular": np.random.randint(1, 6, n),
    "bill_payment_timely": np.random.randint(0, 2, n),
    "digital_payments_freq": np.random.randint(1, 10, n),
    "late_payments_count": np.random.randint(0, 6, n),
    "income_level": np.random.randint(1, 6, n),
    "employment_stability": np.random.randint(1, 5, n),
}

df = pd.DataFrame(data)

# rule-based logic to make dataset realistic
df["risk_score_pattern"] = (
    (df["late_payments_count"] * 2)
    - df["bill_payment_timely"]
    - df["mobile_recharge_regular"]
    - df["employment_stability"]
)

# generate repayment label
df["loan_repaid"] = (df["risk_score_pattern"] < 2).astype(int)

df.drop(columns=["risk_score_pattern"], inplace=True)

df.to_csv("synthetic_alt_data_credit.csv", index=False)

print("Synthetic dataset generated successfully!")
print(df.head())
