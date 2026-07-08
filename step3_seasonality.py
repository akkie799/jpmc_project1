import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)

# Extract year and month as separate columns
df['Year'] = df['Dates'].dt.year
df['Month'] = df['Dates'].dt.month

plt.figure(figsize=(10, 5))

# Draw one line per year, x-axis = month number (1-12)
for year, group in df.groupby('Year'):
    plt.plot(group['Month'], group['Prices'], marker='o', label=str(year))

plt.title('Natural Gas Prices by Month, Grouped by Year')
plt.xlabel('Month')
plt.ylabel('Price')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(True)
plt.tight_layout()
plt.savefig('seasonality_plot.png')
plt.show()