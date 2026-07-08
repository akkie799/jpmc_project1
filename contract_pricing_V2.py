import pandas as pd


def months_between(start_date, end_date):
    """
    Counts whole calendar months between two dates, rounding partial
    months up (any portion of a month counts as a full month of storage).
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    if end_date.day > start_date.day:
        months += 1
    return max(months, 0)


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
    injection_withdrawal_cost_per_unit_volume
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
        (e.g. our predict_price() from the pricing model).
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

    Returns
    -------
    float : the value of the contract in dollars.
    """

    # --- 1. Combine injections and withdrawals into one chronological list of events ---
    # Each event is a dict noting its date, type, and volume.
    events = []
    for date, vol in zip(injection_dates, injection_volumes):
        events.append({"date": pd.to_datetime(date), "type": "inject", "volume": vol})
    for date, vol in zip(withdrawal_dates, withdrawal_volumes):
        events.append({"date": pd.to_datetime(date), "type": "withdraw", "volume": vol})

    # Sort all events by date so we process them in the order they actually happen.
    events.sort(key=lambda e: e["date"])

    total_value = 0.0
    current_volume_in_storage = 0.0
    last_event_date = None

    # --- 2. Walk through events in order, updating cash flows and storage state ---
    for event in events:
        date = event["date"]
        volume = event["volume"]

        # --- Rate limit check ---
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
            # Storage cost is charged as a flat fee per month for using the
            # facility at all while holding gas (not scaled by volume here,
            # matching the task's example of a flat monthly fee).
            total_value -= storage_cost_per_month * num_months

        # --- Apply the event's cash flow and update storage volume ---
        price_on_date = price_lookup(date)

        if event["type"] == "inject":
            current_volume_in_storage += volume
            if current_volume_in_storage > max_storage_volume:
                raise ValueError(
                    f"Injection on {date.date()} exceeds max storage volume "
                    f"({current_volume_in_storage} > {max_storage_volume})."
                )
            total_value -= price_on_date * volume  # cost of buying gas
        else:  # withdraw
            current_volume_in_storage -= volume
            if current_volume_in_storage < 0:
                raise ValueError(
                    f"Withdrawal on {date.date()} exceeds volume currently in storage."
                )
            total_value += price_on_date * volume  # revenue from selling gas

        # --- Injection/withdrawal cost, charged per event ---
        total_value -= injection_withdrawal_cost_per_unit_volume * volume

        last_event_date = date

    return total_value


# --- Test with a simple case matching our single-event example ---
if __name__ == "__main__":
    # A basic price lookup for testing: fixed prices on specific dates.
    def simple_price_lookup(date):
        date = pd.to_datetime(date)
        if date == pd.to_datetime("2024-06-01"):
            return 2.0
        elif date == pd.to_datetime("2024-10-01"):
            return 3.0
        else:
            raise ValueError(f"No price available for {date}")

    value = price_contract(
        injection_dates=["2024-06-01"],
        injection_volumes=[1_000_000],
        withdrawal_dates=["2024-10-01"],
        withdrawal_volumes=[1_000_000],
        price_lookup=simple_price_lookup,
        injection_rate=1_000_000,
        withdrawal_rate=1_000_000,
        max_storage_volume=1_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_unit_volume=0.01
    )
    print(f"Contract value (single event, via generalized function): ${value:,.2f}")

    # --- A more realistic multi-event test case ---
    def multi_price_lookup(date):
        # A small lookup table of prices on specific dates, standing in
        # for our real predict_price() model.
        price_table = {
            "2024-06-01": 2.00,
            "2024-07-01": 2.10,
            "2024-10-01": 3.00,
            "2024-11-01": 3.20,
        }
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        if date_str not in price_table:
            raise ValueError(f"No price available for {date_str}")
        return price_table[date_str]

    multi_value = price_contract(
        injection_dates=["2024-06-01", "2024-07-01"],
        injection_volumes=[500_000, 300_000],
        withdrawal_dates=["2024-10-01", "2024-11-01"],
        withdrawal_volumes=[400_000, 400_000],
        price_lookup=multi_price_lookup,
        injection_rate=500_000,       # max volume per injection event
        withdrawal_rate=400_000,      # max volume per withdrawal event
        max_storage_volume=1_000_000, # facility capacity
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_unit_volume=0.01
    )
    print(f"Contract value (multi-event): ${multi_value:,.2f}")

    # --- Test that constraint violations are correctly caught ---

    # Case 1: Injection volume exceeds the injection rate limit
    try:
        price_contract(
            injection_dates=["2024-06-01"],
            injection_volumes=[600_000],       # exceeds injection_rate below
            withdrawal_dates=["2024-10-01"],
            withdrawal_volumes=[600_000],
            price_lookup=multi_price_lookup,
            injection_rate=500_000,
            withdrawal_rate=600_000,
            max_storage_volume=1_000_000,
            storage_cost_per_month=100_000,
            injection_withdrawal_cost_per_unit_volume=0.01
        )
        print("ERROR: should have raised an exception for exceeding injection rate!")
    except ValueError as e:
        print(f"Correctly caught injection rate violation: {e}")

    # Case 2: Total injected volume exceeds storage capacity
    try:
        price_contract(
            injection_dates=["2024-06-01", "2024-07-01"],
            injection_volumes=[600_000, 600_000],   # totals 1.2M, over capacity
            withdrawal_dates=["2024-11-01"],
            withdrawal_volumes=[1_200_000],
            price_lookup=multi_price_lookup,
            injection_rate=600_000,
            withdrawal_rate=1_200_000,
            max_storage_volume=1_000_000,   # capacity smaller than total injected
            storage_cost_per_month=100_000,
            injection_withdrawal_cost_per_unit_volume=0.01
        )
        print("ERROR: should have raised an exception for exceeding storage capacity!")
    except ValueError as e:
        print(f"Correctly caught storage capacity violation: {e}")

    # Case 3: Withdrawing more than what's currently in storage
    try:
        price_contract(
            injection_dates=["2024-06-01"],
            injection_volumes=[300_000],
            withdrawal_dates=["2024-10-01"],
            withdrawal_volumes=[400_000],   # more than the 300K injected
            price_lookup=multi_price_lookup,
            injection_rate=300_000,
            withdrawal_rate=400_000,
            max_storage_volume=1_000_000,
            storage_cost_per_month=100_000,
            injection_withdrawal_cost_per_unit_volume=0.01
        )
        print("ERROR: should have raised an exception for over-withdrawal!")
    except ValueError as e:
        print(f"Correctly caught over-withdrawal violation: {e}")