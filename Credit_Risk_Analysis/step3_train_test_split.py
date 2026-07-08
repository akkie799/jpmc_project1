import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('../Data/Loan_Data.csv')

# Our independent variables (features) - everything except the ID and the target
feature_columns = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]

X = df[feature_columns]   # inputs
y = df['default']          # target we're trying to predict

# Split into 80% training data, 20% testing data.
# random_state fixes the shuffle so results are reproducible each run.
# stratify=y ensures both the train and test sets keep the same ~18.5%
# default rate, rather than randomly ending up with an unrepresentative split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Default rate in training set:", y_train.mean().round(4))
print("Default rate in test set:", y_test.mean().round(4))