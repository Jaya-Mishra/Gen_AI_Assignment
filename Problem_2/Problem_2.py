# Problem Statement
# A financial institution wants to predict whether a customer will default on a loan before approving it. Early identification of risky customers helps reduce financial loss.
# You are working as a Machine Learning Analyst and must build a classification model using the K-Nearest Neighbors (KNN) algorithm to predict loan default.
# This case introduces:
# •	Mixed feature types
# •	Financial risk interpretation
# •	Class imbalance awareness
# Age,AnnualIncome(lakhs),CreditScore(300-900), LoanAmount(lakhs), LoanTerm(years), EmploymentType, loan(yes/no)
# 28,6.5,720,5,5,Salaried,0
# 45,12,680,10,10,Self-Employed,1
# 35,8,750,6,7,Salaried,0
# 50,15,640,12,15,Self-Employed,1
# 30,7,710,5,5,Salaried,0
# 42,10,660,9,10,Salaried,1
# 26,5.5,730,4,4,Salaried,0
# 48,14,650,11,12,Self-Employed,1
# 38,9,700,7,8,Salaried,0
# 55,16,620,13,15,Self-Employed,1
# Interpretation
# 1.	Identify high-risk customers.
# 2.	What patterns lead to loan default?
# 3.	How do credit score and income influence predictions?
# 4.	Suggest banking policies based on model output.
# 5.	Compare KNN with Decision Trees for this problem.
# 6.	What happens if LoanAmount dominates distance calculation?
# 7.	Should KNN be used in real-time loan approval systems?
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = {
    "Age": [28, 45, 35, 50, 30, 42, 26, 48, 38, 55],
    "AnnualIncome": [6.5, 12, 8, 15, 7, 10, 5.5, 14, 9, 16],
    "CreditScore": [720, 680, 750, 640, 710, 660, 730, 650, 700, 620],
    "LoanAmount": [5, 10, 6, 12, 5, 9, 4, 11, 7, 13],
    "LoanTerm": [5, 10, 7, 15, 5, 10, 4, 12, 8, 15],
    "EmploymentType": [
        "Salaried", "Self-Employed", "Salaried", "Self-Employed",
        "Salaried", "Salaried", "Salaried", "Self-Employed",
        "Salaried", "Self-Employed"
    ],
    "LoanDefault": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print(df)
le = LabelEncoder()
df["EmploymentType"] = le.fit_transform(df["EmploymentType"])
X = df.drop("LoanDefault", axis=1)
y = df["LoanDefault"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("\n===== MODEL EVALUATION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Predict for a New Customer
new_customer = pd.DataFrame({
    "Age": [40],
    "AnnualIncome": [9],
    "CreditScore": [690],
    "LoanAmount": [8],
    "LoanTerm": [10],
    "EmploymentType": ["Salaried"]
})
new_customer["EmploymentType"] = le.transform(new_customer["EmploymentType"])
new_customer_scaled = scaler.transform(new_customer)
prediction = knn.predict(new_customer_scaled)
print("\n===== NEW CUSTOMER PREDICTION =====")
print("Loan Default:", "YES" if prediction[0] == 1 else "NO")

# 10. INTERPRETATION SECTION
print("\n===== INTERPRETATION SECTION =====\n")

# 1.Identify high risk customer
x_full_scaled = scaler.transform(X)
print("\n 1. Identify High Risk Customers")
df["PredictedDefault"] = knn.predict(x_full_scaled)
high_risk_customers = df[df["PredictedDefault"] == 1]
print("High Risk Customers:\n")
print(high_risk_customers)

interpretation = {
    "2. What patterns lead to loan default?": """
Patterns that increase default risk:
- Low credit score → poor repayment history
- High loan amount vs income → financial stress
- Long loan tenure → higher uncertainty
- Self-employed borrowers → irregular income
These factors together raise the probability of default.
""",
    "3. How do credit score and income influence predictions?": """
Credit Score:
- Most influential feature
- Higher score → closer to non-defaulters
- Lower score → closer to defaulters
Annual Income:
- Higher income reduces default risk
- Low income + high loan amount strongly increases default risk
Both heavily affect KNN distance calculations.
""",
    "4. Suggest banking policies based on model output": """
Suggested banking policies:
- Lower interest rates for high-credit-score customers
- Risk premiums for low-credit-score customers
- Loan amount caps based on income levels
- Collateral for high-risk applicants
- Manual review for borderline predictions
- Periodic retraining of the model
""",
    "5. Compare KNN with Decision Trees": """
KNN:
- Low interpretability
- Fast training
- Slow prediction for large datasets
- Poor scalability
- Weak for real-time use
Decision Tree:
- High interpretability
- Moderate training
- Fast prediction
- Good scalability
- Strong for real-time use
Conclusion:
Decision Trees are more suitable for real-world banking systems.
""",
    "6. What happens if LoanAmount dominates distance calculation?": """
If LoanAmount has a much larger numeric range:
- It overpowers other features
- The model relies mainly on loan size
- Predictions become biased and inaccurate
Solution:
Apply feature scaling (StandardScaler or MinMaxScaler).
""",
    "7. Should KNN be used in real-time loan approval systems?": """
Not recommended because:
- Slow predictions for large datasets
- High memory usage
- Poor interpretability
- Sensitive to noisy data
Better alternatives:
- Decision Trees
- Random Forest
- Logistic Regression
- XGBoost
"""
}
for key, value in interpretation.items():
    print(key)
    print(value)
