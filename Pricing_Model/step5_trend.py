import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../Data/Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)
df['DaysSinceStart'] = (df['Dates'] - df['Dates'].min()).dt.days

# Reshape into the 2D array format sklearn expects: (n_samples, n_features)
X = df['DaysSinceStart'].values.reshape(-1, 1)
y = df['Prices'].values

# Fit the linear model
model = LinearRegression()
model.fit(X, y)

print("Slope (price change per day):", model.coef_[0])
print("Intercept (price at day 0):", model.intercept_)

# Predict the trend line for every point we have
df['Trend'] = model.predict(X)

# Plot actual prices vs the fitted trend line
plt.figure(figsize=(10, 5))
plt.plot(df['Dates'], df['Prices'], marker='o', label='Actual Prices')
plt.plot(df['Dates'], df['Trend'], label='Fitted Trend', linewidth=2)
plt.title('Natural Gas Prices with Linear Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('trend_plot.png')
plt.show()