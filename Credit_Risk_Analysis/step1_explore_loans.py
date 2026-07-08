import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.read_csv('../Data/Loan_Data.csv')

# --- Check 1: overall default rate ---
default_rate = df['default'].mean()
print(f"Overall default rate: {default_rate:.2%}")
print(f"Number of defaults: {df['default'].sum()} out of {len(df)}")

# --- Check 2: compare average feature values between defaulters and non-defaulters ---
# groupby splits the data into two groups (default=0, default=1) and .mean()
# gives us the average of every other column within each group.
comparison = df.drop(columns=['customer_id']).groupby('default').mean()
print("\nAverage values by group (0 = no default, 1 = default):")
print(comparison)