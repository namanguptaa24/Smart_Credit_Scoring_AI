import pandas as pd

df = pd.read_csv("synthetic_alt_data_credit.csv")

# remove duplicate records
df.drop_duplicates(inplace=True)

# handle missing values
df.fillna({
    "mobile_recharge_regular": df["mobile_recharge_regular"].median(),
    "digital_payments_freq": df["digital_payments_freq"].median(),
    "income_level": df["income_level"].median(),
    "employment_stability": df["employment_stability"].mode()[0],
}, inplace=True)

# enforce integer types
int_cols = [
    "has_smartphone",
    "mobile_recharge_regular",
    "bill_payment_timely",
    "digital_payments_freq",
    "late_payments_count",
    "income_level",
    "employment_stability"
]

df[int_cols] = df[int_cols].astype(int)

# remove unrealistic extreme values
df = df[(df["late_payments_count"] <= 6) &
        (df["digital_payments_freq"] <= 15)]

df.to_csv("clean_credit_dataset.csv", index=False)

print("Dataset cleaned & saved as clean_credit_dataset.csv")
