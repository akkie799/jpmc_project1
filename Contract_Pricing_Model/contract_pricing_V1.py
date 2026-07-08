def months_between(start_date, end_date):
    """
    Counts the number of calendar months between two dates, treating any
    partial month as a full month (i.e. rounds up).

    Example: June 1 -> October 1 = 4 months (June, July, Aug, Sept occupied,
    withdrawn right at the start of Oct so Oct itself isn't charged).
    Example: June 15 -> October 1 = 4 months (partial June counts as a
    full month of storage).
    """
    import pandas as pd
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Difference in whole year/month terms
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # If the withdrawal falls on a later day-of-month than the injection day,
    # that's an extra partial month tacked on at the end - round it up.
    if end_date.day > start_date.day:
        months += 1

    return max(months, 0)  # never negative


def price_contract_single(
    injection_date,
    withdrawal_date,
    injection_price,
    withdrawal_price,
    volume,
    storage_cost_per_month,
    injection_withdrawal_cost_per_unit_volume
):
    """
    Prices a simple single-injection, single-withdrawal gas storage contract.
    (See docstring from before for parameter details.)
    """
    # --- Revenue and purchase cost ---
    purchase_cost = injection_price * volume
    sale_revenue = withdrawal_price * volume

    # --- Storage cost: billed by whole calendar month, partial months round up ---
    num_months = months_between(injection_date, withdrawal_date)
    total_storage_cost = storage_cost_per_month * num_months

    # --- Injection/withdrawal costs: charged once per event, per unit volume ---
    injection_cost = injection_withdrawal_cost_per_unit_volume * volume
    withdrawal_cost = injection_withdrawal_cost_per_unit_volume * volume

    # --- Net contract value ---
    contract_value = (
        sale_revenue
        - purchase_cost
        - total_storage_cost
        - injection_cost
        - withdrawal_cost
    )

    return contract_value


if __name__ == "__main__":
    value = price_contract_single(
        injection_date="2024-06-01",
        withdrawal_date="2024-10-01",
        injection_price=2.0,
        withdrawal_price=3.0,
        volume=1_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_unit_volume=0.01
    )
    print(f"Contract value: ${value:,.2f}")