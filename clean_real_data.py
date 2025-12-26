import pandas as pd

df = pd.read_csv("application_train.csv")

print("Original shape:", df.shape)

# Drop rows with missing key values
df = df.dropna()

# Remove invalid balances / ages
df = df[df["LIMIT_BAL"] > 0]
df = df[df["AGE"] >= 18]

# Clip extreme unrealistic children count if present
if "CNT_CHILDREN" in df.columns:
    df["CNT_CHILDREN"] = df["CNT_CHILDREN"].clip(upper=5)

# Create repayment outcome (1=repaid, 0=default)
df["loan_repaid"] = (df["default.payment.next.month"] == 0).astype(int)

df.to_csv("creditcard_clean_data.csv", index=False)

print("Clean dataset saved as creditcard_clean_data.csv")
print("Shape after cleaning:", df.shape)
print(df.head())
