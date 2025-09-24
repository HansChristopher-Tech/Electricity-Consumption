import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Setup
os.system("cls" if os.name == "nt" else "clear")
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv(r"C:\Users\Hans Christopher\Documents\DATA ANALYST TOOLS\PYTHON\Electricity Consumption\Results\final_csv.csv")

# Prepare columns
df = df.rename(columns={'Unnamed: 0': 'Day'}, errors='ignore')
df = df.rename(columns={'Date': 'ds', 'Daily_kWh': 'y'}, errors='ignore')

#Transform y to log
df["y_log"] = np.log(df['y'])

# Ensure numeric & datetime
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# Add unique_id for StatsForecast
df['unique_id'] = '1'
# Prepare df for StatsForecast using log values
df_sf = df[['unique_id', 'ds', 'y_log']].rename(columns={'y_log': 'y'})

# Forecast Horizon
horizon = 365 - len(df)

# Define Models

models = [
    AutoARIMA(seasonal=True, season_length=7, alias="SARIMA"),
    AutoETS(season_length=7, alias="ETS")
]

# Fit and Forecast on log-transformed data
sf = StatsForecast(models=models, freq="D", n_jobs=-1)
sf.fit(df=df_sf)  # use df_sf with y_log
future_preds_log = sf.predict(h=horizon)

# Invert log transformation to original scale
future_preds = future_preds_log.copy()
future_preds['SARIMA'] = np.exp(future_preds_log['SARIMA'])
future_preds['ETS'] = np.exp(future_preds_log['ETS'])

# Add mild noise for realism
std_y = df['y'].std()  # still use original y for std
np.random.seed(3)
future_preds['SARIMA_noisy'] = future_preds['SARIMA'] + np.random.normal(0, 0.5*std_y, size=len(future_preds))
future_preds['ETS_noisy'] = future_preds['ETS'] + np.random.normal(0, 0.5*std_y, size=len(future_preds))

# Optional: clip negative values after noise
future_preds['SARIMA_noisy'] = future_preds['SARIMA_noisy'].clip(lower=0)
future_preds['ETS_noisy'] = future_preds['ETS_noisy'].clip(lower=0)

#Generate New Dataframes
#Everything!
df_final_predictions = pd.concat([df, future_preds], axis=0).reset_index()
df_final_predictions = df_final_predictions.rename(columns={"index": "Day"}, errors="ignore")

#Noisy SARIMA
df_sarima_noise = future_preds.drop(columns=["SARIMA", "ETS", "ETS_noisy"])
df_sarima_noise = df_sarima_noise.rename(columns={"SARIMA_noisy": "y"}, errors="ignore")
df_sarima_final = pd.concat([df, df_sarima_noise], axis=0).reset_index()

#Noisy ETS
df_ets_noise = future_preds.drop(columns=["SARIMA", "ETS", "SARIMA_noisy"])
df_ets_noise = df_ets_noise.rename(columns={"ETS_noisy": "y"}, errors="ignore")
df_ets_final = pd.concat([df, df_ets_noise], axis=0).reset_index()

#Save CSV's
df_final_predictions.to_csv(r"C:\Users\Hans Christopher\Documents\DATA ANALYST TOOLS\PYTHON\Electricity Consumption\Results\final_Predictons.csv")
print("CSV's Saved")

# Plot Results as Subplots
def plotter():
    fig, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True)

    # Set locator to one tick per month
    months = mdates.MonthLocator()  
    month_fmt = mdates.DateFormatter('%b')  # Jan, Feb, Mar ...

    # SARIMA Forecast
    axes[0].plot(df_sarima_final['ds'], df_sarima_final['y'], color='blue', label='Noisy SARIMA Forecast')
    axes[0].set_title('SARIMA Forecast')
    axes[0].set_ylabel('Daily kWh')
    axes[0].legend()
    axes[0].xaxis.set_major_locator(months)
    axes[0].xaxis.set_major_formatter(month_fmt)

    # ETS Forecast
    axes[1].plot(df_ets_final['ds'], df_ets_final['y'], color='green', label='Noisy ETS Forecast')
    axes[1].set_title('ETS Forecast')
    axes[1].set_ylabel('Daily kWh')
    axes[1].legend()
    axes[1].xaxis.set_major_locator(months)
    axes[1].xaxis.set_major_formatter(month_fmt)

    plt.show()

def compute_metrics(y_true, y_pred):
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return mae_val, rmse_val, mape_val

def cross_validation_metrics(sf, df, horizon=30, n_windows=8, step_size=7):
    # Run cross-validation
    cv_df = sf.cross_validation(
    df=df_sf,      # your training dataframe
    h=30,          # forecast horizon
    n_windows=8,   # number of rolling windows
    step_size=7,   # window step
    refit=True
)

    # Prepare a table
    metrics_table = []

    models = ['SARIMA', 'ETS']
    for model in models:
        # y_true and y_pred for all windows
        y_true_all = cv_df['y'].values
        y_pred_all = cv_df[model].values

        mae_val, rmse_val, mape_val = compute_metrics(y_true_all, y_pred_all)
        metrics_table.append({
            'Model': model,
            'MAE': mae_val,
            'RMSE': rmse_val,
            'MAPE (%)': mape_val
        })

    return pd.DataFrame(metrics_table)

# --------------------
# Usage
# --------------------
cv_results = cross_validation_metrics(sf, df_sf, horizon=30)
print(cv_results)