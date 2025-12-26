import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

# load dataset to reuse schema
df = pd.read_csv("synthetic_alt_data_credit.csv")
X = df.drop("loan_repaid", axis=1)
y = df["loan_repaid"]

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X, y)

joblib.dump(model, "credit_model.pkl")

print("Model saved as credit_model.pkl")
