import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("../data/cibil_loan_dataset.csv")
df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

# Train model
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

model = RandomForestClassifier()
model.fit(X, y)

# UI
st.title("Loan Prediction System (CIBIL Score)")

income = st.number_input("Income")
loan = st.number_input("Loan Amount")
cibil = st.slider("CIBIL Score", 300, 900)
credit = st.selectbox("Credit History", [0, 1])

if st.button("Predict"):
    prediction = model.predict([[income, loan, cibil, credit]])

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
