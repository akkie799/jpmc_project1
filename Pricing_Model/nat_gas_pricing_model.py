"""
Natural Gas Price Estimation Model
------------------------------------
Fits a trend + seasonal model to historical monthly natural gas prices
(Oct 2020 - Sep 2024) and provides a function to estimate the price
on any date, including up to one year beyond the historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1. Load and prepare the data ---

# Read the raw CSV. At this point 'Dates' is just text (e.g. "10/31/20"),
# not a real date type yet.
df = pd.read_csv('../Data/Nat_Gas.csv')

# Convert 'Dates' from text into actual datetime objects, using the exact
# format we know the data is in (month/day/2-digit-year). Being explicit
# about the format avoids pandas guessing incorrectly.
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Sort chronologically and reset the row index (0, 1, 2, ...) so nothing
# is out of order for the time-series logic that follows.
df = df.sort_values('Dates').reset_index(drop=True)

# Store the earliest date - this becomes our reference point ("day 0").
start_date = df['Dates'].min()

# Convert every date into "number of days since start_date". Regression
# models need numbers, not date objects, so this gives us a clean numeric
# time variable to fit against.
df['DaysSinceStart'] = (df['Dates'] - start_date).dt.days


# --- 2. Build features: linear trend + annual seasonal cycle (sin/cos) ---

# Average number of days in a year (accounts for leap years). This sets
# the period of our seasonal wave to repeat once every 12 months.
ANNUAL_PERIOD = 365.25

def build_features(days_since_start):
    """
    Given one or more "days since start" values, build the feature matrix
    the model expects: [day_count, sin(annual cycle), cos(annual cycle)].

    Using both sin and cos (rather than just one) lets the fitted wave
    peak at ANY point in the year, not just at a fixed position - the
    regression figures out the right combination automatically.

    This function is used both when fitting the model and when predicting,
    so the two never accidentally get out of sync.
    """
    days_since_start = np.asarray(days_since_start)
    return np.column_stack([
        days_since_start,                                       # captures long-term trend
        np.sin(2 * np.pi * days_since_start / ANNUAL_PERIOD),   # seasonal wave, sine part
        np.cos(2 * np.pi * days_since_start / ANNUAL_PERIOD)    # seasonal wave, cosine part
    ])

# Build the training features (X) and targets (y) from our historical data.
X = build_features(df['DaysSinceStart'].values)
y = df['Prices'].values


# --- 3. Fit the model ---

# LinearRegression finds the best-fit weights for each feature (day count,
# sin, cos) plus an intercept, minimizing squared error against actual
# prices. It's still "linear" regression because it's a weighted sum of
# features, even though the sin/cos features themselves are curved.
model = LinearRegression()
model.fit(X, y)

# Check how well the fitted model explains the historical data.
# R^2 close to 1.0 = explains almost all the variation.
# R^2 close to 0.0 = no better than guessing the average price.
r2 = r2_score(y, model.predict(X))
print(f"Model R^2 on historical data: {r2:.4f}")


# --- 4. Prediction function: takes any date, returns estimated price ---

def predict_price(date_input):
    """
    Estimate the natural gas price on a given date.

    Parameters
    ----------
    date_input : str or datetime-like
        The date to estimate the price for, e.g. '2025-03-15'.
        Works for historical dates (interpolation) and dates up to
        ~1 year beyond the last known data point (extrapolation).

    Returns
    -------
    float : estimated price, rounded to 2 decimal places
    """
    # Accepts strings, datetime objects, etc. - pandas parses flexibly.
    date = pd.to_datetime(date_input)

    # Convert the requested date into the same "days since start" scale
    # the model was trained on.
    days_since_start = (date - start_date).days

    # Build the same 3 features used in training (day count, sin, cos)
    # for this single date, then ask the model to predict.
    features = build_features([days_since_start])
    predicted_price = model.predict(features)[0]

    return round(predicted_price, 2)


# --- 5. Demonstration: plot historical data + model fit + extrapolation ---
if __name__ == "__main__":

    last_date = df['Dates'].max()

    # Build one continuous run of daily dates spanning the ENTIRE range:
    # from the very first historical date through one year past the last
    # historical date. This lets us draw the model's fitted line as a
    # single unbroken curve - matching actual data where we have it, and
    # continuing seamlessly into the extrapolated future.
    all_dates = pd.date_range(
        start=df['Dates'].min(),
        end=last_date + pd.Timedelta(days=365),
        freq='D'
    )
    all_fitted_prices = [predict_price(d) for d in all_dates]

    plt.figure(figsize=(11, 5))

    # The real, actual monthly price points we were given.
    plt.plot(df['Dates'], df['Prices'], marker='o', label='Historical Prices (actual)')

    # The model's continuous fitted curve: overlays the historical range
    # AND extends one year into the future, using the exact same formula
    # throughout (no seam between "fit" and "forecast").
    plt.plot(all_dates, all_fitted_prices, label='Model Fit + Extrapolation',
              linewidth=2, color='orange')

    # Mark exactly where known data ends and projection begins.
    plt.axvline(last_date, color='gray', linestyle='--', alpha=0.6,
                label='End of Historical Data')

    plt.title('Natural Gas Prices: Historical Data + 1-Year Extrapolation')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('final_forecast_plot.png')
    plt.show()

    # A few example lookups, demonstrating the function works for both
    # interpolation (within historical range) and extrapolation (beyond it).
    example_dates = ['2021-06-15', '2023-12-25', '2024-09-30',
                      '2025-01-15', '2025-09-30']
    print("\nExample price estimates:")
    for d in example_dates:
        print(f"  {d} -> {predict_price(d)}")
            