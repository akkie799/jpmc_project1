# JPMorgan Chase & Co. — Quantitative Research Virtual Experience

This repo contains my work from the JPMorgan Chase Quantitative Research virtual experience program (via Forage). It covers four tasks spanning commodity price forecasting, derivative-style contract pricing, credit risk modeling, and a quantization/dynamic programming problem.

> **Note:** This is coursework from Forage's JPMorgan Chase virtual experience program, not actual work product from JPMorgan Chase. It's intended to demonstrate applied skills in Python, statistical modeling, and quantitative finance.

---

## Project 1: Natural Gas Price Forecasting
**Folder:** `Pricing_Model/`
**Script:** `nat_gas_pricing_model.py`

Builds a model to estimate the price of natural gas on any given date, using ~4 years of monthly historical price snapshots (Oct 2020 – Sep 2024).

- Fits a combined **linear trend + seasonal (sine/cosine) regression** to capture both the long-term upward price trend and the recurring within-year seasonal pattern (higher in winter, lower in summer — consistent with heating demand).
- Achieves an **R² of 0.93** on historical data.
- Exposes a `predict_price(date)` function that works for both **interpolation** (any historical date) and **extrapolation** (up to ~1 year beyond the last known data point).

**To run:**
```bash
python3 Pricing_Model/nat_gas_pricing_model.py
```
Requires `Nat_Gas.csv` in `Data/`.

---

## Project 2: Gas Storage Contract Pricing
**Folder:** `Contract_Pricing_Model/`
**Script:** `gas_storage_contract_pricer.py`

Prices a natural gas storage contract (buy low, store, sell high) given one or more injection and withdrawal dates.

- Generalizes to **multiple injection/withdrawal events** in any order, using a chronological event-based cash flow ledger.
- Accounts for **purchase/sale cash flows**, **storage costs** (billed by whole calendar month), and **injection/withdrawal fees**.
- Enforces physical constraints: **injection/withdrawal rate limits** and **maximum storage capacity**, raising clear errors on invalid contracts (e.g. withdrawing more than what's in storage).
- Uses the Project 1 price model (`predict_price`) as its price source, so it can price contracts using both historical and extrapolated future prices.
- Includes an optional `verbose=True` mode that prints a full line-by-line cash flow breakdown for auditability.

**To run:**
```bash
python3 Contract_Pricing_Model/gas_storage_contract_pricer.py
```
Requires `Nat_Gas.csv` in `Data/`.

---

## Project 3: Loan Default Probability & Expected Loss
**Folder:** `Credit_Risk_Analysis/`
**Script:** `loan_default_expected_loss_model.py`

Builds a model to estimate a borrower's probability of default (PD) from their loan/credit characteristics, then converts that into an expected loss estimate.

- Trains and compares two models: **logistic regression** (primary) and a **decision tree** (comparison), using features including credit lines outstanding, total debt, income, years employed, and FICO score.
- Logistic regression achieves **ROC AUC = 1.00** and **99.9% accuracy** on held-out test data (decision tree: 0.9996 AUC, 99.5% accuracy) — worth noting this near-perfect separability likely reflects how cleanly structured this particular sample dataset is (`credit_lines_outstanding` alone almost perfectly separates the two classes), rather than something expected from noisier real-world production data.
- Exposes `predict_expected_loss(...)`, which takes a borrower's details and loan amount and returns both the predicted PD and the expected loss, using:
  ```
  Expected Loss = PD × (1 − Recovery Rate) × Loan Amount
  ```
  with a recovery rate of 10%, as specified by the task.

**To run:**
```bash
python3 Credit_Risk_Analysis/loan_default_expected_loss_model.py
```
Requires `Loan_Data.csv` in `Data/`.

---

## Project 4: FICO Score Bucketing (Quantization)
**Folder:** `FICO_Bucketing/`
**Script:** `fico_bucketing_model.py`

Maps a borrower's FICO score to a discrete rating (1 = best credit, higher = worse), for use as a categorical input to downstream default-prediction models. Implements and compares two bucketing strategies:

- **Mean squared error (MSE) bucketing** — via 1D KMeans clustering, grouping numerically similar FICO scores without reference to default outcomes.
- **Log-likelihood bucketing** — via a custom **dynamic programming** algorithm, choosing boundaries that maximize the log-likelihood of the observed default pattern across buckets. This is the primary, general-purpose approach, since it directly optimizes for separating default risk rather than just numerical closeness of scores.
- The log-likelihood method produces a sharper risk separation (worst bucket: 66% default rate) than the MSE method (worst bucket: 51% default rate), at the cost of less evenly-sized buckets.
- Both approaches are reusable for any number of buckets, so the mapping generalizes to future datasets.
