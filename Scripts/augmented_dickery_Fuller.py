import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
import os
import time

# Import Data
df = pd.read_csv(r"C:\Users\Hans Christopher\Documents\DATA ANALYST TOOLS\PYTHON\Electricity Consumption\household_power_consumption.csv")

# Combine Date and Time into a proper datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Convert to numeric (some datasets have "?" or strings)
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Drop missing values (optional)
df = df.dropna(subset=['Global_active_power'])

# Set Datetime as index
df.set_index('Datetime', inplace=True)

# Resample into daily kWh:
# Global_active_power is in kW, each row is 1 minute → divide sum by 60
daily_power = df['Global_active_power'].resample('D').sum() / 60

# Reset index to get a clean DataFrame
daily_power = daily_power.reset_index()
daily_power.columns = ['Date', 'Daily_kWh']

"""
#Plot Original Data
plt.plot(daily_power["Date"], daily_power["Daily_kWh"])
plt.xlabel("Date")
plt.ylabel("kWh")
plt.show()
"""

#Performing the test

# Set significance level
alpha = 0.05  
max_diff = 5   # safety limit (to avoid infinite loop)

series = daily_power["Daily_kWh"]
diff_count = 0

while True:
    # Perform ADF test
    result = adfuller(series.dropna())
    p_value = result[1]

    print(f"\n=== Differencing Round: {diff_count} ===")
    print("ADF Statistic:", result[0])
    print("p-value:", p_value)
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value}")

    # Always state the hypotheses
    print("\nHypotheses of the ADF Test:")
    print("   H0: The time series is NON-STATIONARY (has a unit root)")
    print("   H1: The time series is STATIONARY")

    if p_value < alpha:
        print(f"> p-value = {p_value:.4f} <= {alpha} → Reject H0 → Series is STATIONARY ✅")
        break
    else:
        if diff_count >= max_diff:
            print("> Reached max differencing limit, stopping.")
            break
        
        print(f"> p-value = {p_value:.4f} > {alpha} → Fail to reject H0 → Series is NON-STATIONARY ❌")
        print("> Applying differencing...")
        time.sleep(2)

        # Apply differencing
        series = series.diff().dropna()
        diff_count += 1

# Final differenced series is stored in `series`
print(f"\nFinal differencing order used: d = {diff_count}")

# Store back to dataframe for reference
daily_power[f"Diff_{diff_count}"] = series

# Plot original vs differenced
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(daily_power['Date'], daily_power['Daily_kWh'], label="Original")
plt.title("Original Daily kWh")
plt.subplot(1,2,2)
plt.plot(daily_power.loc[series.index, 'Date'], series, 
         label=f"{diff_count} Differences", color="orange")
plt.title(f"{diff_count} Differences")
plt.show()

#Save as CSV
daily_power.to_csv(r"C:\Users\Hans Christopher\Documents\DATA ANALYST TOOLS\PYTHON\Electricity Consumption\Results\Charts")
print("CSV's Saved")