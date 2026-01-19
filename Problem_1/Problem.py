# A company wants to predict employee productivity scores to improve workforce planning and training programs. You are hired as a Data Scientist to build a multivariate linear regression model that predicts an employee’s Productivity Score based on multiple work-related factors.
# Experience (yrs),Training Hours,Working Hours,Projects,Productivity Score
# 2,40,38,3,62
# 5,60,42,6,78
# 1,20,35,2,55
# 8,80,45,8,88
# 4,50,40,5,72
# 10,90,48,9,92
# 3,30,37,4,65
# 6,70,44,7,82
# 7,75,46,7,85
# 2,25,36,3,60
# Interpretation
# •	Which factor most strongly impacts productivity?
# •	How does training affect productivity?
# •	Should the company increase training hours or working hours?
# •	What happens if Working Hours increase beyond optimal limits?
# •	Can productivity ever decrease with more experience?
# •	How would you detect overfitting in this model?
# •	Suggest one new feature to improve prediction accuracy.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
 
data = {
    "Experience": [2, 5, 1, 8, 4, 10, 3, 6, 7, 2],
    "TrainingHours": [40, 60, 20, 80, 50, 90, 30, 70, 75, 25],
    "WorkingHours": [38, 42, 35, 45, 40, 48, 37, 44, 46, 36],
    "Projects": [3, 6, 2, 5, 5, 9, 4, 7, 8, 3],
    "ProductivityScore": [62, 78, 55, 88, 72, 92, 65, 82, 85, 60]
}
 
df = pd.DataFrame(data)
 
X = df[["Experience", "TrainingHours", "WorkingHours", "Projects"]]
y = df["ProductivityScore"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)
 
print("\nIntercept:", round(model.intercept_, 3))
print("\nCoefficients (sorted by impact):\n", coeff_df)
 
#   Which factor most strongly impacts productivity?
most_impactful = coeff_df.iloc[0]
print(f"\nMost impactful factor: {most_impactful['Feature']} "
      f"(Coefficient = {round(most_impactful['Coefficient'], 3)})")
 
#   How does training affect productivity?
training_coef = coeff_df[coeff_df["Feature"] == "TrainingHours"]["Coefficient"].values[0]
if training_coef > 0:
    print(f"TrainingHours has a POSITIVE effect on productivity "
          f"(+{round(training_coef, 3)} per additional hour).")
else:
    print(f"TrainingHours has a NEGATIVE effect on productivity "
          f"({round(training_coef, 3)} per additional hour).")
 
#   Should company increase training or working hours?
working_coef = coeff_df[coeff_df["Feature"] == "WorkingHours"]["Coefficient"].values[0]
 
if training_coef > working_coef:
    print("Recommendation: Increase TRAINING hours rather than WORKING hours.")
else:
    print("Recommendation: Increase WORKING hours rather than TRAINING hours.")
 
# Predictions & evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
 
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
 
print("\nR² (Train):", round(r2_train, 3))
print("R² (Test):", round(r2_test, 3))
print("Mean Squared Error (Test):", round(mse_test, 3))
 
# How to detect overfitting?
if r2_train - r2_test > 0.15:
    print("Possible overfitting detected (train R² much higher than test R²).")
else:
    print("No strong evidence of overfitting.")
 
# Can productivity decrease with more experience?
experience_coef = coeff_df[coeff_df["Feature"] == "Experience"]["Coefficient"].values[0]
 
if experience_coef < 0:
    print("Productivity may DECREASE with more experience (negative coefficient).")
else:
    print("Productivity INCREASES with more experience (positive coefficient).")
 
# What happens if working hours increase beyond limits?
print("\nNote: Linear regression assumes productivity increases linearly with working hours.")
print("In reality, productivity may plateau or decline due to burnout.")
print("A polynomial or nonlinear model would be better for this analysis.")
 
# Suggested new feature
print("\nSuggested new feature to improve accuracy: WorkLifeBalance")
 
# Predict for a new employee
new_employee = np.array([[5, 50, 42, 6]])
predicted_productivity = model.predict(new_employee)
 
print("\nPredicted Productivity Score for new employee:",
      round(predicted_productivity[0], 2))
 
