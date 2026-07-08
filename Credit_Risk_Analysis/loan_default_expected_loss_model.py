"""
Loan Default Probability & Expected Loss Model
------------------------------------------------
Trains a model to predict a borrower's probability of default (PD) from
their loan/credit characteristics, then uses that PD - together with a
specified recovery rate - to compute the expected loss on a loan.

Two modeling approaches are fit and compared (logistic regression and a
decision tree); logistic regression is used as the primary model for the
final predict_expected_loss() function, since it performed marginally
better and is more standard/interpretable for credit risk use cases.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ============================================================
# PART A: Load data and prepare train/test split
# ============================================================

df = pd.read_csv('../Data/Loan_Data.csv')

FEATURE_COLUMNS = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]

X = df[FEATURE_COLUMNS]
y = df['default']

# 80/20 train/test split. stratify=y keeps the ~18.5% default rate
# consistent between both sets. random_state fixes the split so results
# are reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# PART B: Train logistic regression (primary model)
# ============================================================

# Logistic regression is sensitive to feature scale (income is in the
# tens of thousands, credit_lines_outstanding is 0-5), so we standardize
# every feature to mean 0 / std 1 before fitting. The scaler is fit only
# on training data to avoid leaking test-set information.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

logreg_pred = logreg_model.predict(X_test_scaled)
logreg_pred_proba = logreg_model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# PART C: Train decision tree (comparison model)
# ============================================================

# Decision trees split on raw thresholds (e.g. "fico_score < 610"), so
# feature scale doesn't affect their splits - trained on unscaled features.
# max_depth is capped to prevent the tree from memorizing the training
# data instead of generalizing.
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)
tree_pred_proba = tree_model.predict_proba(X_test)[:, 1]

# ============================================================
# PART D: Compare both models on the held-out test set
# ============================================================

def print_model_performance(name, y_true, y_pred, y_pred_proba):
    print(f"\n{name} performance on test set:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 score:  {f1_score(y_true, y_pred):.4f}")
    print(f"  ROC AUC:   {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")


# ============================================================
# PART E: Expected loss function (the actual deliverable)
# ============================================================

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
    probability of default (PD) using the trained logistic regression
    model, then returns the expected loss on the loan:

        Expected Loss = PD * (1 - Recovery Rate) * Loan Amount

    Parameters
    ----------
    credit_lines_outstanding : int
        Number of credit lines the borrower currently has outstanding.
    loan_amt_outstanding : float
        The outstanding loan amount. Also used as the exposure at default
        (the amount at risk if the borrower defaults).
    total_debt_outstanding : float
        Borrower's total outstanding debt across all sources.
    income : float
        Borrower's income.
    years_employed : int
        Number of years the borrower has been employed.
    fico_score : int
        Borrower's FICO credit score.

    Returns
    -------
    dict with:
        'probability_of_default' : float, between 0 and 1
        'expected_loss' : float, in dollars
    """
    # Build a single-row DataFrame with matching column names/order so the
    # scaler and model treat this new borrower identically to training data.
    borrower_features = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score
    }])

    borrower_scaled = scaler.transform(borrower_features)

    # predict_proba returns [P(no default), P(default)] - we want column 1.
    pd_estimate = logreg_model.predict_proba(borrower_scaled)[0, 1]

    expected_loss = pd_estimate * (1 - RECOVERY_RATE) * loan_amt_outstanding

    return {
        'probability_of_default': float(round(pd_estimate, 4)),
        'expected_loss': float(round(expected_loss, 2))
    }


# ============================================================
# PART F: Run comparisons and example predictions
# ============================================================
if __name__ == "__main__":

    # --- Show learned feature importances (logistic regression coefficients) ---
    coefficients = pd.Series(logreg_model.coef_[0], index=FEATURE_COLUMNS)
    print("Logistic regression feature coefficients (scaled), sorted by influence:")
    print(coefficients.sort_values(key=abs, ascending=False))

    # --- Compare both models' performance ---
    print_model_performance("Logistic Regression", y_test, logreg_pred, logreg_pred_proba)
    print_model_performance("Decision Tree", y_test, tree_pred, tree_pred_proba)

    # --- Example predictions ---
    safe_borrower = predict_expected_loss(
        credit_lines_outstanding=0,
        loan_amt_outstanding=5000,
        total_debt_outstanding=5000,
        income=70000,
        years_employed=5,
        fico_score=700
    )
    print("\nSafe-looking borrower:", safe_borrower)

    risky_borrower = predict_expected_loss(
        credit_lines_outstanding=5,
        loan_amt_outstanding=5000,
        total_debt_outstanding=20000,
        income=70000,
        years_employed=1,
        fico_score=580
    )
    print("Risky-looking borrower:", risky_borrower)