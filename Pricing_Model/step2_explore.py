import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../Data/Nat_Gas.csv')

# Convert the Dates column from text to real datetime objects
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Sort by date just to be safe (in case the CSV isn't in order)
df = df.sort_values('Dates').reset_index(drop=True)

# Confirm the conversion worked
print(df.dtypes)
print(df.head())

# Plot the raw price series over time
plt.figure(figsize=(10, 5))
plt.plot(df['Dates'], df['Prices'], marker='o')
plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.tight_layout()
plt.savefig('price_plot.png')
plt.show()