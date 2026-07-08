import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('../Data/Loan_Data.csv')
fico_scores = df['fico_score'].values

NUM_BUCKETS = 5

# KMeans expects a 2D array (n_samples, n_features), even though we only
# have 1 feature (fico_score). reshape(-1, 1) adds that second dimension.
X = fico_scores.reshape(-1, 1)

kmeans = KMeans(n_clusters=NUM_BUCKETS, random_state=42, n_init=10)
kmeans.fit(X)

# Each score gets assigned a cluster label (0 to NUM_BUCKETS-1), but these
# labels are arbitrary - cluster "0" doesn't necessarily mean "worst" or
# "best" scores. We need to figure out the actual ordering ourselves.
cluster_centers = kmeans.cluster_centers_.flatten()
print("Raw cluster centers (unordered):", sorted(cluster_centers))

# To get proper bucket BOUNDARIES (not just cluster centers), we find the
# midpoint between each pair of adjacent cluster centers, once sorted.
sorted_centers = np.sort(cluster_centers)
boundaries = [(sorted_centers[i] + sorted_centers[i + 1]) / 2 for i in range(len(sorted_centers) - 1)]

print("Bucket boundaries:", boundaries)

# Boundaries define bucket edges. We add -infinity and +infinity as the
# outermost edges so every possible FICO score (even ones outside our
# training data's observed range) falls into some bucket.
full_boundaries = [-np.inf] + boundaries + [np.inf]

def fico_to_rating(fico_score):
    """
    Maps a FICO score to a rating using the MSE-optimal bucket boundaries.
    Lower rating = better credit score (Rating 1 = best bucket).

    Parameters
    ----------
    fico_score : int or float

    Returns
    -------
    int : rating from 1 (best) to NUM_BUCKETS (worst)
    """
    # Find which bucket this score falls into. np.digitize returns the
    # index of the bucket (1-indexed by default when using these boundaries),
    # counting from the LOWEST FICO bucket upward.
    bucket_index = np.digitize(fico_score, full_boundaries)

    # bucket_index=1 means "lowest FICO bucket" (worst credit), but we want
    # Rating 1 to mean BEST credit. So we flip the numbering:
    # highest bucket_index (best FICO) -> Rating 1
    rating = (NUM_BUCKETS + 1) - bucket_index
    return int(rating)


# --- Test the mapping ---
if __name__ == "__main__":
    test_scores = [420, 560, 610, 650, 700, 750, 840]
    for score in test_scores:
        print(f"FICO {score} -> Rating {fico_to_rating(score)}")

# --- Check: does this bucketing separate default risk? ---
df['rating'] = df['fico_score'].apply(fico_to_rating)

risk_by_rating = df.groupby('rating').agg(
    count=('default', 'size'),
    default_rate=('default', 'mean')
)
print("\nDefault rate by rating (MSE-based buckets):")
print(risk_by_rating.sort_index())