"""
Natural Gas Storage Contract Pricing Model
--------------------------------------------
Prices a gas storage contract with one or more injection and withdrawal
dates, accounting for purchase/sale cash flows, storage costs, and
injection/withdrawal fees. Uses the trend+seasonal price model built
previously (predict_price) as the source of buy/sell prices on any date.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ============================================================
# PART A: Price model (from the earlier task)
# ============================================================

# --- Load and prepare historical price data ---
df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)

# Reference point: "day 0" of our time axis
start_date = df['Dates'].min()
df['DaysSinceStart'] = (df['Dates'] - start_date).dt.days

# Average days per year, used to set the seasonal cycle's period
ANNUAL_PERIOD = 365.25

def build_features(days_since_start):
    """
    Builds the [day_count, sin(annual cycle), cos(annual cycle)] feature
    matrix used by the price model. Shared between fitting and predicting
    so the two never drift out of sync.
    """
    days_since_start = np.asarray(days_since_start)
    return np.column_stack([
        days_since_start,
        np.sin(2 * np.pi * days_since_start / ANNUAL_PERIOD),
        np.cos(2 * np.pi * days_since_start / ANNUAL_PERIOD)
    ])

# --- Fit the trend + seasonal model once, at import time ---
X = build_features(df['DaysSinceStart'].values)
y = df['Prices'].values

price_model = LinearRegression()
price_model.fit(X, y)


def predict_price(date_input):
    """
    Estimate the natural gas price on a given date.

    Works for historical dates (interpolation) and dates up to ~1 year
    beyond the last known data point (extrapolation), using the same
    fitted trend + seasonal formula throughout.
    """
    date = pd.to_datetime(date_input)
    days_since_start = (date - start_date).days
    features = build_features([days_since_start])
    return round(price_model.predict(features)[0], 2)


# ============================================================
# PART B: Contract pricing model (this task)
# ============================================================

def months_between(start_date, end_date):
    """
    Counts whole calendar months between two dates, rounding any partial
    month up. Any portion of a month spent in storage is billed as a
    full month (standard for facility rental billing).
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Difference expressed in whole year/month terms
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # If withdrawal falls on a later day-of-month than injection,
    # that's an extra partial month tacked on at the end - round up.
    if end_date.day > start_date.day:
        months += 1

    return max(months, 0)  # never negative


def price_contract(
    injection_dates,
    injection_volumes,
    withdrawal_dates,
    withdrawal_volumes,
    price_lookup,
    injection_rate,
    withdrawal_rate,
    max_storage_volume,
    storage_cost_per_month,
    injection_withdrawal_cost_per_unit_volume,
    verbose=False
):
    """
    Prices a natural gas storage contract with multiple injection and
    withdrawal dates.

    Parameters
    ----------
    injection_dates : list of dates
        Dates on which gas is bought and injected into storage.
    injection_volumes : list of float
        Volume injected on each corresponding injection date.
    withdrawal_dates : list of dates
        Dates on which gas is withdrawn and sold.
    withdrawal_volumes : list of float
        Volume withdrawn on each corresponding withdrawal date.
    price_lookup : function
        A function taking a date and returning the price on that date
        (e.g. predict_price from Part A).
    injection_rate : float
        Maximum volume that can be injected in a single event.
    withdrawal_rate : float
        Maximum volume that can be withdrawn in a single event.
    max_storage_volume : float
        Maximum volume the facility can hold at any point in time.
    storage_cost_per_month : float
        Fixed fee charged per month (or partial month) gas is held.
    injection_withdrawal_cost_per_unit_volume : float
        Cost charged per unit volume, each time gas is injected OR withdrawn.
    verbose : bool, default False
        If True, prints a line-by-line breakdown of every cash flow as
        it's processed, for transparency/auditability.

    Returns
    -------
    total_value : float
        The value of the contract in dollars.
    breakdown : list of dict
        A chronological log of every cash flow contributing to the total,
        each with a date, description, and amount.
    """

    # --- 1. Combine injections and withdrawals into one chronological event list ---
    events = []
    for date, vol in zip(injection_dates, injection_volumes):
        events.append({"date": pd.to_datetime(date), "type": "inject", "volume": vol})
    for date, vol in zip(withdrawal_dates, withdrawal_volumes):
        events.append({"date": pd.to_datetime(date), "type": "withdraw", "volume": vol})
    events.sort(key=lambda e: e["date"])

    total_value = 0.0
    current_volume_in_storage = 0.0
    last_event_date = None
    breakdown = []  # running, human-readable log of every cash flow

    def log(date, description, amount):
        """Records one cash flow line and updates the running total."""
        nonlocal total_value
        total_value += amount
        breakdown.append({"date": date.date(), "description": description, "amount": amount})
        if verbose:
            print(f"  {date.date()}  {description:<45} {amount:>15,.2f}")

    # --- 2. Walk through events in order, updating cash flows and storage state ---
    for event in events:
        date = event["date"]
        volume = event["volume"]

        # --- Rate limit checks ---
        if event["type"] == "inject" and volume > injection_rate:
            raise ValueError(
                f"Injection volume {volume} on {date.date()} exceeds max injection rate {injection_rate}."
            )
        if event["type"] == "withdraw" and volume > withdrawal_rate:
            raise ValueError(
                f"Withdrawal volume {volume} on {date.date()} exceeds max withdrawal rate {withdrawal_rate}."
            )

        # --- Storage cost accrued since the last event ---
        # Charge for whatever volume was sitting in storage during the
        # gap between the previous event and this one.
        if last_event_date is not None and current_volume_in_storage > 0:
            num_months = months_between(last_event_date, date)
            if num_months > 0:
                log(
                    date,
                    f"Storage cost ({num_months} month(s), {current_volume_in_storage:,.0f} held)",
                    -storage_cost_per_month * num_months
                )

        # --- Apply the event's cash flow and update storage volume ---
        price_on_date = price_lookup(date)

        if event["type"] == "inject":
            current_volume_in_storage += volume
            if current_volume_in_storage > max_storage_volume:
                raise ValueError(
                    f"Injection on {date.date()} exceeds max storage volume "
                    f"({current_volume_in_storage} > {max_storage_volume})."
                )
            log(date, f"Buy {volume:,.0f} units @ ${price_on_date:.2f}", -price_on_date * volume)
        else:  # withdraw
            current_volume_in_storage -= volume
            if current_volume_in_storage < 0:
                raise ValueError(
                    f"Withdrawal on {date.date()} exceeds volume currently in storage."
                )
            log(date, f"Sell {volume:,.0f} units @ ${price_on_date:.2f}", price_on_date * volume)

        # --- Injection/withdrawal fee, charged per event ---
        fee_label = "Injection fee" if event["type"] == "inject" else "Withdrawal fee"
        log(date, fee_label, -injection_withdrawal_cost_per_unit_volume * volume)

        last_event_date = date

    if verbose:
        print(f"  {'':<57} {'-' * 15}")
        print(f"  {'TOTAL CONTRACT VALUE':<57} {total_value:>15,.2f}")

    return total_value, breakdown


# ============================================================
# PART C: Example usage, using the real price model
# ============================================================
if __name__ == "__main__":

    # Example: client wants to inject in summer 2024, withdraw in winter,
    # using our actual fitted price model to look up prices automatically
    # (no need to hardcode buy/sell prices - predict_price supplies them).
    # Withdrawal dates deliberately extend past the historical data's end
    # (Sep 30, 2024) to exercise the extrapolated part of the price model.
    value, breakdown = price_contract(
        injection_dates=["2024-06-01", "2024-07-01"],
        injection_volumes=[500_000, 300_000],
        withdrawal_dates=["2024-12-01", "2025-01-15"],
        withdrawal_volumes=[400_000, 400_000],
        price_lookup=predict_price,
        injection_rate=500_000,
        withdrawal_rate=400_000,
        max_storage_volume=1_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_unit_volume=0.01,
        verbose=True
    )
    print(f"\nFinal contract value: ${value:,.2f}")