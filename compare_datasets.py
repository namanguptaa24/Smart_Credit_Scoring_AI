import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

def train_eval(path):
    df = pd.read_csv(path)
    X = df.drop("loan_repaid", axis=1)
    y = df["loan_repaid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    return accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)

synthetic_acc, synthetic_auc = train_eval("synthetic_alt_data_credit.csv")
real_acc, real_auc = train_eval("altdata_creditcard_final.csv")

print("\nDataset Performance Comparison")
print("--------------------------------")
print("Synthetic Dataset:")
print("Accuracy:", round(synthetic_acc,3), "AUC:", round(synthetic_auc,3))

print("\nReal Dataset:")
print("Accuracy:", round(real_acc,3), "AUC:", round(real_auc,3))
