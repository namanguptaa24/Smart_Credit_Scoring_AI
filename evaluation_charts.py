import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("synthetic_alt_data_credit.csv")
X = df.drop("loan_repaid", axis=1)
y = df["loan_repaid"]

model = joblib.load("credit_model.pkl")

from sklearn.metrics import ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix — Credit Risk Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve — Credit Risk Model")
plt.tight_layout()
plt.savefig("roc_curve.png")

print("Charts generated successfully.")
