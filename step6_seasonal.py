import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)
df['DaysSinceStart'] = (df['Dates'] - df['Dates'].min()).dt.days

X = df['DaysSinceStart'].values.reshape(-1, 1)
y = df['Prices'].values

model = LinearRegression()
model.fit(X, y)
df['Trend'] = model.predict(X)

# Residual = actual price minus trend = the leftover "wiggle"
df['Residual'] = df['Prices'] - df['Trend']

# Look at average residual per calendar month across all years
df['Month'] = df['Dates'].dt.month
monthly_avg_residual = df.groupby('Month')['Residual'].mean()

print(monthly_avg_residual)

plt.figure(figsize=(10, 5))
plt.bar(monthly_avg_residual.index, monthly_avg_residual.values)
plt.title('Average Seasonal Residual by Month')
plt.xlabel('Month')
plt.ylabel('Average Residual (Price - Trend)')
plt.xticks(range(1, 13))
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('seasonal_residual_plot.png')
plt.show()