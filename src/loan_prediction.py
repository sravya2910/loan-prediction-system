import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/cibil_loan_dataset.csv")

# Convert target variable
df["Loan_Status"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
