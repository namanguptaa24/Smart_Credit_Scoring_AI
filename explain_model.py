import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("credit_model.pkl")
df = pd.read_csv("synthetic_alt_data_credit.csv")

X = df.drop("loan_repaid", axis=1)

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values()

feat_imp.plot(kind="barh")
plt.title("Feature Importance — Smart Credit Scoring AI")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png")

print("Feature importance chart generated.")
print(feat_imp.sort_values(ascending=False))
