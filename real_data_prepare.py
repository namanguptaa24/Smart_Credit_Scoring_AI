import pandas as pd

# Load your uploaded dataset
df = pd.read_csv("application_train.csv")

print("\nOriginal Shape :", df.shape)
print("\nColumns Detected:\n", df.columns.tolist())

# ---------- Keep useful columns ----------

cols = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",

    # repayment status history (behaviour signal)
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",

    # bill statements
    "BILL_AMT1","BILL_AMT2","BILL_AMT3",
    "BILL_AMT4","BILL_AMT5","BILL_AMT6",

    # repayment amounts
    "PAY_AMT1","PAY_AMT2","PAY_AMT3",
    "PAY_AMT4","PAY_AMT5","PAY_AMT6",

    # target column
    "default.payment.next.month"
]

available_cols = [c for c in cols if c in df.columns]
df = df[available_cols]

print("\nUsing columns:\n", available_cols)
print("\nShape after selection :", df.shape)

# ---------- Remove invalid rows ----------

df = df.dropna()
df = df[df["LIMIT_BAL"] > 0]
df = df[df["AGE"] >= 18]

# ---------- Create cleaner target label ----------

df["loan_repaid"] = (df["default.payment.next.month"] == 0).astype(int)

print("\nTarget distribution (1 = repaid, 0 = default):")
print(df["loan_repaid"].value_counts())

# ---------- Save Clean Subset ----------

df.to_csv("creditcard_clean_subset.csv", index=False)

print("\nSaved cleaned dataset -> creditcard_clean_subset.csv")
print("\nPreview:\n", df.head())
