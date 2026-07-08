import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('../Data/Loan_Data.csv')

feature_columns = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]

X = df[feature_columns]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale the features ---
# Logistic regression is sensitive to the SCALE of each feature. Notice our
# features have very different ranges: fico_score is in the hundreds, income
# is in the tens of thousands, credit_lines_outstanding is 0-5. Without
# scaling, features with naturally larger numbers can dominate the fit even
# if they're not actually more important. StandardScaler rescales every
# feature to have mean 0 and standard deviation 1, putting them on equal footing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Fit the logistic regression model ---
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- Inspect the fitted coefficients ---
# Each coefficient tells us how strongly (and in which direction) that
# feature pushes the prediction toward "default" once scaled to the same
# footing as the others. Larger absolute value = bigger influence.
coefficients = pd.Series(model.coef_[0], index=feature_columns)
print("Feature coefficients (scaled):")
print(coefficients.sort_values(key=abs, ascending=False))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# --- Generate predictions on the held-out test set ---
y_pred = model.predict(X_test_scaled)                    # hard 0/1 predictions
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]   # actual probability of default

# --- Evaluate ---
print("\nModel performance on test set:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))