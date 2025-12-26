import pandas as pd

df = pd.read_csv("creditcard_clean_data.csv")

# ---------- Income capacity proxy ----------
df["income_level"] = pd.qcut(
    df["LIMIT_BAL"],
    q=5,
    labels=[1,2,3,4,5]
).astype(int)

# ---------- Late payment behaviour ----------
pay_cols = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
df["late_payments_count"] = (df[pay_cols] > 0).sum(axis=1)

# ---------- Bill payment responsibility ----------
df["bill_payment_timely"] = (df["late_payments_count"] == 0).astype(int)

# ---------- Digital activity frequency ----------
pay_amt_cols = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
df["digital_payments_freq"] = (df[pay_amt_cols] > 0).sum(axis=1).clip(upper=10)

# ---------- Spending stability indicator ----------
bill_cols = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
df["spending_stability"] = df[bill_cols].std(axis=1).fillna(0)

# ---------- Employment stability (age bucket proxy) ----------
df["employment_stability"] = pd.cut(
    df["AGE"],
    bins=[20,25,30,40,50,100],
    labels=[1,2,3,4,5]
).astype(int)

# ---------- Digital inclusion proxy ----------
df["has_smartphone"] = (df["EDUCATION"] <= 3).astype(int)

# ---------- Final feature frame ----------
selected = [
    "has_smartphone",
    "bill_payment_timely",
    "digital_payments_freq",
    "late_payments_count",
    "income_level",
    "employment_stability",
    "spending_stability",
    "loan_repaid"
]

df_final = df[selected]
df_final.to_csv("altdata_creditcard_final.csv", index=False)

print("Alt-data feature dataset saved as altdata_creditcard_final.csv")
print(df_final.head())
