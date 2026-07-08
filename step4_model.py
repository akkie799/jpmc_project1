import pandas as pd
import numpy as np

df = pd.read_csv('Nat_Gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df = df.sort_values('Dates').reset_index(drop=True)

# Days since the first data point - our numeric "time" variable
df['DaysSinceStart'] = (df['Dates'] - df['Dates'].min()).dt.days

print(df.head())
print(df[['Dates', 'DaysSinceStart', 'Prices']].tail())