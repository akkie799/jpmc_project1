"""
FICO Score Bucketing (Quantization) Model
--------------------------------------------
Maps a borrower's FICO score to a discrete rating (1 = best credit,
higher = worse credit), for use as a categorical input to Charlie's
downstream default-prediction model.

Two approaches are implemented and compared:

1. Mean Squared Error (MSE) bucketing - via 1D KMeans clustering.
   Groups FICO scores that are numerically close together, without
   reference to actual default outcomes.

2. Log-likelihood bucketing - via dynamic programming.
   Chooses boundaries that best explain the observed pattern of defaults,
   directly optimizing for how distinctly each bucket's default rate
   stands out. This is the general, data-driven approach requested by
   the task, and is used as the primary rating map.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# ============================================================
# PART A: Load and prepare data
# ============================================================

df = pd.read_csv('../Data/Loan_Data.csv')

# Collapse to one row per unique FICO score, with count (n) and number
# of defaults (k) at that score - this is what the log-likelihood/DP
# method operates on, and keeps that part of the algorithm fast.
agg = df.groupby('fico_score').agg(
    n=('default', 'size'),
    k=('default', 'sum')
).reset_index().sort_values('fico_score').reset_index(drop=True)


# ============================================================
# PART B: MSE-based bucketing (1D KMeans clustering)
# ============================================================

def fit_mse_boundaries(num_buckets):
    """
    Finds bucket boundaries that minimize mean squared error between
    each FICO score and its bucket's average, via 1D KMeans clustering.

    Returns a sorted list of boundary values (length num_buckets - 1).
    """
    fico_scores = df['fico_score'].values
    X = fico_scores.reshape(-1, 1)

    kmeans = KMeans(n_clusters=num_buckets, random_state=42, n_init=10)
    kmeans.fit(X)

    sorted_centers = np.sort(kmeans.cluster_centers_.flatten())
    boundaries = [
        (sorted_centers[i] + sorted_centers[i + 1]) / 2
        for i in range(len(sorted_centers) - 1)
    ]
    return boundaries


# ============================================================
# PART C: Log-likelihood bucketing (dynamic programming)
# ============================================================

def bucket_log_likelihood(n_total, k_total):
    """
    Log-likelihood contribution of a single bucket, assuming a constant
    default probability p = k_total / n_total within the bucket.

    Formula: k*log(p) + (n-k)*log(1-p)

    Guards against log(0) in the degenerate cases where p = 0 or p = 1.
    """
    if n_total == 0:
        return 0.0

    p = k_total / n_total

    if p == 0:
        return (n_total - k_total) * np.log(1 - p)
    if p == 1:
        return k_total * np.log(p)

    return k_total * np.log(p) + (n_total - k_total) * np.log(1 - p)


# Prefix sums over the aggregated table, so any range's (n, k) totals
# can be computed in O(1) instead of re-summing each time.
_cumulative_n = np.concatenate([[0], np.cumsum(agg['n'].values)])
_cumulative_k = np.concatenate([[0], np.cumsum(agg['k'].values)])


def _get_bucket_stats(start_idx, end_idx):
    """(n_total, k_total) for aggregated rows in [start_idx, end_idx)."""
    n_total = _cumulative_n[end_idx] - _cumulative_n[start_idx]
    k_total = _cumulative_k[end_idx] - _cumulative_k[start_idx]
    return n_total, k_total


def fit_loglikelihood_boundaries(num_buckets):
    """
    Uses dynamic programming to find the FICO score boundaries that
    maximize total log-likelihood when split into `num_buckets` buckets.

    dp[i][k] = best total log-likelihood using the first i unique FICO
    rows (sorted ascending), split into exactly k buckets. Built up from
    smaller subproblems: dp[i][k] considers every possible previous cut
    point j, combining the already-solved dp[j][k-1] with the
    log-likelihood of one new bucket spanning rows [j, i).

    Returns
    -------
    boundaries : list of float
        FICO score values separating each bucket.
    best_score : float
        Total log-likelihood achieved by this optimal split.
    """
    num_rows = len(agg)

    dp = np.full((num_rows + 1, num_buckets + 1), -np.inf)
    split_point = np.full((num_rows + 1, num_buckets + 1), -1, dtype=int)

    # Base case: k=1 bucket covering the first i rows.
    for i in range(1, num_rows + 1):
        n_i, k_i = _get_bucket_stats(0, i)
        dp[i][1] = bucket_log_likelihood(n_i, k_i)
        split_point[i][1] = 0

    # Build up solutions for k=2..num_buckets buckets.
    for k in range(2, num_buckets + 1):
        for i in range(k, num_rows + 1):
            for j in range(k - 1, i):
                if dp[j][k - 1] == -np.inf:
                    continue
                n_bucket, k_bucket = _get_bucket_stats(j, i)
                candidate_score = dp[j][k - 1] + bucket_log_likelihood(n_bucket, k_bucket)
                if candidate_score > dp[i][k]:
                    dp[i][k] = candidate_score
                    split_point[i][k] = j

    # Reconstruct the chosen cut points by walking backward through
    # split_point, from the full problem down to the base case.
    boundaries_idx = []
    i, k = num_rows, num_buckets
    while k > 1:
        j = split_point[i][k]
        boundaries_idx.append(j)
        i, k = j, k - 1
    boundaries_idx.reverse()

    boundaries = [
        (agg['fico_score'].iloc[idx - 1] + agg['fico_score'].iloc[idx]) / 2
        for idx in boundaries_idx
    ]

    return boundaries, dp[num_rows][num_buckets]


# ============================================================
# PART D: Build rating map functions from a set of boundaries
# ============================================================

def make_rating_function(boundaries):
    """
    Given a list of bucket boundaries, returns a function that maps a
    FICO score to a rating, where Rating 1 = best credit (highest FICO
    bucket) and higher ratings = worse credit.
    """
    full_boundaries = [-np.inf] + sorted(boundaries) + [np.inf]
    num_buckets = len(full_boundaries) - 1

    def fico_to_rating(fico_score):
        bucket_index = np.digitize(fico_score, full_boundaries)
        rating = (num_buckets + 1) - bucket_index
        return int(rating)

    return fico_to_rating


# ============================================================
# PART E: Fit both approaches and compare
# ============================================================
if __name__ == "__main__":
    NUM_BUCKETS = 5

    # --- MSE approach ---
    mse_boundaries = fit_mse_boundaries(NUM_BUCKETS)
    fico_to_rating_mse = make_rating_function(mse_boundaries)
    df['rating_mse'] = df['fico_score'].apply(fico_to_rating_mse)

    print("MSE-based boundaries:", mse_boundaries)
    print("\nDefault rate by rating (MSE-based buckets):")
    print(df.groupby('rating_mse').agg(
        count=('default', 'size'), default_rate=('default', 'mean')
    ).sort_index())

    # --- Log-likelihood / DP approach ---
    ll_boundaries, best_ll_score = fit_loglikelihood_boundaries(NUM_BUCKETS)
    fico_to_rating_ll = make_rating_function(ll_boundaries)
    df['rating_ll'] = df['fico_score'].apply(fico_to_rating_ll)

    print("\nLog-likelihood-based boundaries:", ll_boundaries)
    print(f"Best total log-likelihood: {best_ll_score:.4f}")
    print("\nDefault rate by rating (log-likelihood-based buckets):")
    print(df.groupby('rating_ll').agg(
        count=('default', 'size'), default_rate=('default', 'mean')
    ).sort_index())

    # --- Example usage: the rating map Charlie would actually use ---
    print("\nExample FICO -> Rating mappings (log-likelihood method):")
    for score in [420, 560, 610, 650, 700, 750, 840]:
        print(f"  FICO {score} -> Rating {fico_to_rating_ll(score)}")

        