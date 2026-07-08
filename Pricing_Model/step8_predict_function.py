import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Load and prepare data ---
df = pd.read_csv('../Data/Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)

start_date = df['Dates'].min()
df['DaysSinceStart'] = (df['Dates'] - start_date).dt.days

# --- Fit the model once ---
days = df['DaysSinceStart'].values
X = np.column_stack([
    days,
    np.sin(2 * np.pi * days / 365.25),
    np.cos(2 * np.pi * days / 365.25)
])
y = df['Prices'].values

model = LinearRegression()
model.fit(X, y)

# --- The reusable prediction function ---
def predict_price(date_input):
    """
    Takes a date (string like '2025-03-15' or a datetime object)
    and returns the estimated natural gas price on that date.
    """
    date = pd.to_datetime(date_input)
    days_since_start = (date - start_date).days

    features = np.array([[
        days_since_start,
        np.sin(2 * np.pi * days_since_start / 365.25),
        np.cos(2 * np.pi * days_since_start / 365.25)
    ]])

    predicted_price = model.predict(features)[0]
    return round(predicted_price, 2)


# --- Try it out ---
if __name__ == "__main__":
    test_dates = [
        "2021-06-15",   # within historical range
        "2023-12-25",   # within historical range
        "2024-09-30",   # last date in our data
        "2025-01-15",   # extrapolated, ~3 months out
        "2025-09-30",   # extrapolated, full year out
    ]

    for d in test_dates:
        print(f"{d} -> estimated price: {predict_price(d)}")

    # --- Check model quality ---
    from sklearn.metrics import r2_score
    predictions_on_training_data = model.predict(X)
    r2 = r2_score(y, predictions_on_training_data)
    print(f"\nR^2 score: {r2:.4f}")
        