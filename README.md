🧠 AI for Financial Inclusion — Smart Credit Scoring for the Unbanked

Behaviour-driven credit risk scoring system that uses alternative data signals instead of traditional credit history — enabling fair & inclusive lending for individuals without formal credit records.

This project demonstrates how AI can assist micro-finance, NBFCs and digital lenders in making responsible & explainable loan decisions.

🎯 Problem Statement

Millions of individuals & small businesses in India lack formal credit history.
Traditional scoring systems (like CIBIL) fail to evaluate them — forcing them to rely on:

- informal money lenders

- extremely high interest rates
  
- debt-trap situations

Yet many are actually creditworthy based on real behaviour patterns such as:

- timely bill payments

- mobile recharge consistency

- digital payment activity

- spending stability

- employment regularity

This project shows how AI + alternative behaviour signals can help assess credit risk more fairly.

🚀 Solution Overview

We develop a Machine Learning–based Credit Risk Prediction System using:

✔ Alternative-data inspired behavioural features

✔ Real credit dataset + synthetic dataset

✔ Explainable AI scoring (SHAP)

✔ Human-in-loop loan-officer validation

✔ Fairness & bias awareness

✔ Streamlit-based scoring dashboard


The model generates:

- Default probability

- Risk category (Low / Medium / High)

- Decision support recommendation

- Feature-impact explanation

📊 Datasets Used
✅ Real Dataset — UCI Credit Card Default

Used for real-world validation & model training.

Features engineered into behavioural signals such as:

- late payment count

- bill payment timeliness

- digital payment frequency

- spending stability

- employment stability proxy

- income-level bucket

🧩 Behaviour-Driven Features

| Feature               | Meaning                  |
| --------------------- | ------------------------ |
| bill_payment_timely   | repayment discipline     |
| late_payments_count   | default risk indicator   |
| digital_payments_freq | financial participation  |
| income_level          | affordability proxy      |
| employment_stability  | livelihood stability     |
| spending_stability    | money-flow consistency   |
| has_smartphone        | digital inclusion signal |

🛠 Tech Stack

Machine Learning

- Python

- Pandas / NumPy

- Scikit-learn

- Random Forest Classifier

- SMOTE (optional)

- ROC-AUC evaluation

Explainable AI

- SHAP

- Feature importance visualization

Frontend / App

- Streamlit dashboard

- Officer review panel

- Applicant history tracking

▶️ How to Run

1. Install dependencies:

pip install streamlit shap pandas numpy scikit-learn joblib


2. Train model (first run):

python train_real_model.py


3. Run app:

streamlit run app_credit_ui_pro.py


4. App runs at:

http://localhost:8501
