import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("synthetic_alt_data_credit.csv")

X = df.drop("loan_repaid", axis=1)
y = df["loan_repaid"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Model Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC Score:", round(roc_auc_score(y_test, y_prob), 3))
