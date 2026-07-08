import pandas as pd
import numpy as np

df = pd.read_csv('../Data/Loan_Data.csv')

# Collapse the 10,000 individual borrowers down to one row per UNIQUE
# FICO score, with the total count and total defaults at that score.
agg = df.groupby('fico_score').agg(
    n=('default', 'size'),      # how many borrowers have this exact FICO score
    k=('default', 'sum')        # how many of them defaulted
).reset_index()

agg = agg.sort_values('fico_score').reset_index(drop=True)

print(agg.head(10))
print("...")
print(agg.tail(10))
print(f"\nNumber of unique FICO scores: {len(agg)}")
print(f"Total borrowers (sanity check): {agg['n'].sum()}")
print(f"Total defaults (sanity check): {agg['k'].sum()}")

def bucket_log_likelihood(n_total, k_total):
    """
    Computes the log-likelihood contribution of a single bucket, given:
    n_total : total number of borrowers in the bucket
    k_total : total number of defaults in the bucket

    Formula: k*log(p) + (n-k)*log(1-p), where p = k/n.

    Returns a very negative number (effectively -infinity) for degenerate
    cases where p=0 or p=1, since log(0) is undefined - in practice this
    means "this bucket is impossible to score" rather than "perfect".
    """
    if n_total == 0:
        return 0.0  # an empty bucket contributes nothing

    p = k_total / n_total

    # log(0) is mathematically undefined; guard against p=0 or p=1
    # (a bucket with zero defaults, or where everyone defaulted).
    if p == 0 or p == 1:
        # Treat these as contributing 0 defaults*log(0) safely by only
        # counting the well-defined term.
        if k_total == 0:
            return (n_total - k_total) * np.log(1 - p) if p < 1 else 0.0
        else:
            return k_total * np.log(p) if p > 0 else 0.0

    return k_total * np.log(p) + (n_total - k_total) * np.log(1 - p)


# --- Precompute cumulative sums for fast range queries ---
# To quickly get "total n and total k between row i and row j" without
# re-summing every time, we precompute running totals (prefix sums).
cumulative_n = np.concatenate([[0], np.cumsum(agg['n'].values)])
cumulative_k = np.concatenate([[0], np.cumsum(agg['k'].values)])

def get_bucket_stats(start_idx, end_idx):
    """
    Given a range of rows [start_idx, end_idx) in the aggregated table,
    returns (n_total, k_total) for that range using prefix sums - O(1)
    instead of re-summing the range each time.
    """
    n_total = cumulative_n[end_idx] - cumulative_n[start_idx]
    k_total = cumulative_k[end_idx] - cumulative_k[start_idx]
    return n_total, k_total


# --- Quick test ---
if __name__ == "__main__":
    n_test, k_test = get_bucket_stats(0, 50)
    print(f"First 50 unique scores: n={n_test}, k={k_test}, p={k_test/n_test:.4f}")
    print(f"Log-likelihood of this bucket: {bucket_log_likelihood(n_test, k_test):.4f}")

def find_optimal_boundaries(num_buckets):
    """
    Uses dynamic programming to find the FICO score boundaries that
    maximize total log-likelihood when split into `num_buckets` buckets.

    Returns
    -------
    boundaries : list of float
        The FICO score values that separate each bucket.
    best_score : float
        The total log-likelihood achieved by this optimal split.
    """
    num_rows = len(agg)  # number of unique FICO scores (374 in our case)

    # dp[i][k] = best total log-likelihood using the first i rows, split
    # into exactly k buckets. Initialize everything to -infinity, since
    # we haven't computed anything yet (and want any real computed value
    # to beat this default).
    dp = np.full((num_rows + 1, num_buckets + 1), -np.inf)

    # split_point[i][k] = the "j" (previous cut point) that achieved the
    # best value for dp[i][k]. We need this to reconstruct the actual
    # boundaries afterward, not just know the best SCORE.
    split_point = np.full((num_rows + 1, num_buckets + 1), -1, dtype=int)

    # Base case: using the first i rows, split into exactly 1 bucket,
    # is just the log-likelihood of that whole range as one big bucket.
    for i in range(1, num_rows + 1):
        n_i, k_i = get_bucket_stats(0, i)
        dp[i][1] = bucket_log_likelihood(n_i, k_i)
        split_point[i][1] = 0  # the single bucket starts at row 0

    # Fill in the table for 2, 3, ..., num_buckets buckets.
    for k in range(2, num_buckets + 1):
        for i in range(k, num_rows + 1):  # need at least k rows to make k buckets
            # Try every possible position j for the previous cut point:
            # rows [0, j) form the first k-1 buckets (already optimally
            # solved and stored in dp[j][k-1]), and rows [j, i) form the
            # k-th (newest) bucket.
            for j in range(k - 1, i):
                if dp[j][k - 1] == -np.inf:
                    continue  # this earlier state was never reachable, skip it

                n_bucket, k_bucket = get_bucket_stats(j, i)
                candidate_score = dp[j][k - 1] + bucket_log_likelihood(n_bucket, k_bucket)

                if candidate_score > dp[i][k]:
                    dp[i][k] = candidate_score
                    split_point[i][k] = j

    # --- Reconstruct the actual boundary positions from split_point ---
    boundaries_idx = []
    i, k = num_rows, num_buckets
    while k > 1:
        j = split_point[i][k]
        boundaries_idx.append(j)
        i, k = j, k - 1

    boundaries_idx.reverse()

    # Convert row indices into actual FICO score boundary VALUES: the
    # boundary sits between the score at (index-1) and the score at (index).
    boundaries = [
        (agg['fico_score'].iloc[idx - 1] + agg['fico_score'].iloc[idx]) / 2
        for idx in boundaries_idx
    ]

    best_score = dp[num_rows][num_buckets]
    return boundaries, best_score


# --- Test with 5 buckets ---
if __name__ == "__main__":
    boundaries, best_score = find_optimal_boundaries(5)
    print("Optimal boundaries (log-likelihood method):", boundaries)
    print("Best total log-likelihood:", best_score)   

full_boundaries_ll = [-np.inf] + boundaries + [np.inf]
NUM_BUCKETS_LL = 5

def fico_to_rating_loglikelihood(fico_score):
    """
    Maps a FICO score to a rating using the log-likelihood-optimal
    bucket boundaries. Lower rating = better credit score.
    """
    bucket_index = np.digitize(fico_score, full_boundaries_ll)
    rating = (NUM_BUCKETS_LL + 1) - bucket_index
    return int(rating)


# --- Compare default rate by rating, log-likelihood method ---
df['rating_ll'] = df['fico_score'].apply(fico_to_rating_loglikelihood)

risk_by_rating_ll = df.groupby('rating_ll').agg(
    count=('default', 'size'),
    default_rate=('default', 'mean')
)
print("\nDefault rate by rating (log-likelihood-based buckets):")
print(risk_by_rating_ll.sort_index())     