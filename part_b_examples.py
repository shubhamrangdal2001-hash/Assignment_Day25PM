"""
Part B - Regression and Classification Examples
One example each with clear input, output, and use case.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# ─────────────────────────────────────────────
# REGRESSION EXAMPLE: Predict student exam score
# ─────────────────────────────────────────────
print("=" * 50)
print("REGRESSION: Predicting Student Exam Score")
print("=" * 50)

# Input: hours studied, attendance percentage
# Output: predicted exam score (continuous number)
# Use case: help teachers identify students who need support

reg_data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attendance_pct': [60, 65, 70, 72, 75, 80, 82, 85, 90, 95],
    'exam_score':    [35, 42, 50, 55, 60, 68, 72, 78, 85, 91]
}

reg_df = pd.DataFrame(reg_data)
X_reg = reg_df[['hours_studied', 'attendance_pct']]
y_reg = reg_df['exam_score']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print(f"Predicted scores: {y_pred.round(2)}")
print(f"Actual scores:    {list(y_test)}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print()

# Predict for a new student
new_student = pd.DataFrame({'hours_studied': [5], 'attendance_pct': [78]})
predicted_score = reg_model.predict(new_student)
print(f"New student (5 hrs study, 78% attendance) => Predicted score: {predicted_score[0]:.1f}")
print()

# ─────────────────────────────────────────────
# CLASSIFICATION EXAMPLE: Predict loan approval
# ─────────────────────────────────────────────
print("=" * 50)
print("CLASSIFICATION: Predicting Loan Approval")
print("=" * 50)

# Input: income, credit score, loan amount
# Output: approved (1) or rejected (0)
# Use case: bank automates initial loan screening

cls_data = {
    'income_lpa':    [3, 5, 8, 2, 6, 10, 4, 7, 1.5, 9],
    'credit_score':  [600, 650, 750, 580, 700, 780, 610, 720, 550, 760],
    'loan_amount':   [5, 8, 15, 3, 10, 20, 6, 12, 2, 18],
    'approved':      [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
}

cls_df = pd.DataFrame(cls_data)
X_cls = cls_df[['income_lpa', 'credit_score', 'loan_amount']]
y_cls = cls_df['approved']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

cls_model = LogisticRegression()
cls_model.fit(X_train2, y_train2)
y_pred2 = cls_model.predict(X_test2)

print(f"Predicted (0=rejected, 1=approved): {list(y_pred2)}")
print(f"Actual:                             {list(y_test2)}")
print(f"Accuracy: {accuracy_score(y_test2, y_pred2) * 100:.1f}%")
print()

# Predict for a new applicant
new_applicant = pd.DataFrame({
    'income_lpa': [5.5],
    'credit_score': [690],
    'loan_amount': [9]
})
result = cls_model.predict(new_applicant)
print(f"New applicant (5.5L income, 690 score, 9L loan) => {'Approved' if result[0] == 1 else 'Rejected'}")
