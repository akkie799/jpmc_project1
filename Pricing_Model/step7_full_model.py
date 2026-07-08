import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../Data/Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)
df['DaysSinceStart'] = (df['Dates'] - df['Dates'].min()).dt.days

days = df['DaysSinceStart'].values

# Build the feature columns: days (for trend), sin and cos (for seasonality)
X = np.column_stack([
    days,
    np.sin(2 * np.pi * days / 365.25),
    np.cos(2 * np.pi * days / 365.25)
])
y = df['Prices'].values

model = LinearRegression()
model.fit(X, y)

df['FullModelFit'] = model.predict(X)

print("Coefficients [slope, sin_coef, cos_coef]:", model.coef_)
print("Intercept:", model.intercept_)

plt.figure(figsize=(10, 5))
plt.plot(df['Dates'], df['Prices'], marker='o', label='Actual Prices')
plt.plot(df['Dates'], df['FullModelFit'], label='Trend + Seasonal Fit', linewidth=2)
plt.title('Natural Gas Prices: Actual vs Full Model Fit')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('full_model_plot.png')
plt.show()
