import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Load data and train the model (same as before) ---
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# --- Comparison model: Decision Tree ---
# Note: decision trees don't require feature scaling (they split on raw
# thresholds like "fico_score < 610", so scale doesn't affect the splits),
# so we train it on the unscaled features directly.
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

tree_pred_proba = tree_model.predict_proba(X_test)[:, 1]
tree_pred = tree_model.predict(X_test)

print("Decision Tree performance on test set:")
print(f"Accuracy: {accuracy_score(y_test, tree_pred):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, tree_pred_proba):.4f}")

RECOVERY_RATE = 0.10  # 10%, as specified by the task


def predict_expected_loss(
    credit_lines_outstanding,
    loan_amt_outstanding,
    total_debt_outstanding,
    income,
    years_employed,
    fico_score
):
    """
    Given a borrower's details and their loan amount, predicts the
    probability of default (PD) and returns the expected loss on the loan.

    Expected Loss = PD * (1 - Recovery Rate) * Loan Amount

    Parameters
    ----------
    credit_lines_outstanding : int
    loan_amt_outstanding : float
        The outstanding loan amount - this is also treated as the
        exposure at default (the amount at risk if the borrower defaults).
    total_debt_outstanding : float
    income : float
    years_employed : int
    fico_score : int

    Returns
    -------
    dict with:
        'probability_of_default' : float, between 0 and 1
        'expected_loss' : float, in dollars
    """
    # Build a single-row DataFrame with the same column names/order used
    # in training, so the scaler and model treat it identically.
    borrower_features = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score
    }])

    # Apply the SAME scaling that was fit on the training data
    borrower_scaled = scaler.transform(borrower_features)

    # Predict probability of default (column 1 = probability of class "1"/default)
    pd_estimate = model.predict_proba(borrower_scaled)[0, 1]

    expected_loss = pd_estimate * (1 - RECOVERY_RATE) * loan_amt_outstanding

    return {
        'probability_of_default': float(round(pd_estimate, 4)),
        'expected_loss': float(round(expected_loss, 2))
    }


# --- Test with a few sample borrowers ---
if __name__ == "__main__":
    # A "safe-looking" borrower: few credit lines, low debt, good fico score
    safe_borrower = predict_expected_loss(
        credit_lines_outstanding=0,
        loan_amt_outstanding=5000,
        total_debt_outstanding=5000,
        income=70000,
        years_employed=5,
        fico_score=700
    )
    print("Safe-looking borrower:", safe_borrower)

    # A "risky-looking" borrower: many credit lines, high debt, poor fico score
    risky_borrower = predict_expected_loss(
        credit_lines_outstanding=5,
        loan_amt_outstanding=5000,
        total_debt_outstanding=20000,
        income=70000,
        years_employed=1,
        fico_score=580
    )
    print("Risky-looking borrower:", risky_borrower)