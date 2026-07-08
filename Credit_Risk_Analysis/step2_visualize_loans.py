import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/Loan_Data.csv')

features_to_plot = ['credit_lines_outstanding', 'total_debt_outstanding', 'fico_score', 'income']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()  # turn the 2x2 grid into a flat list of 4, easier to loop over

for i, feature in enumerate(features_to_plot):
    ax = axes[i]
    # Separate the feature's values by default status, then plot both as histograms
    # on the same axes so we can visually compare their distributions.
    df[df['default'] == 0][feature].hist(ax=ax, bins=30, alpha=0.5, label='No Default')
    df[df['default'] == 1][feature].hist(ax=ax, bins=30, alpha=0.5, label='Default')
    ax.set_title(feature)
    ax.legend()

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()