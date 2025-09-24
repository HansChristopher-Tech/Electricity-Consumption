# ⚡ Electrical Consumption Forecasting 📊

## 🔹 Overview
This project focuses on **forecasting daily electricity consumption** using time series analysis. It leverages **ARIMA/SARIMA** and **ETS** models to predict future electricity usage, helping visualize trends and anticipate energy demand. The project is built using **Python**, `statsforecast`, and other data analysis libraries.

---

## 🛠 Features
- 📈 Historical electricity consumption analysis  
- 🔮 Forecasting with:
  - **SARIMA (Seasonal ARIMA)**
  - **ETS (Exponential Smoothing)**
- 🌟 Noise simulation for more realistic predictions  
- 📊 Evaluation metrics: **MAE**, **RMSE**, **MAPE**  
- 📉 Plots of historical and forecasted consumption  
- 📝 Log transformation to handle non-stationarity  

---

## ⚡ How it Works
1. **Load historical electricity consumption data**  
2. **Preprocess data**:
   - Convert dates to `datetime`  
   - Convert consumption to numeric  
   - Add `unique_id` for `StatsForecast`  
3. Apply **log transformation** to avoid negative predictions  
4. Fit **SARIMA** and **ETS** models using `statsforecast`  
5. Forecast future consumption for a defined horizon (e.g., 365 days)  
6. Add mild **random noise** to make predictions more realistic  
7. Clip negative values to avoid impossible electricity usage  
8. Plot results using **matplotlib**, with **monthly x-axis formatting**  
9. Evaluate forecasts using metrics like **MAE**, **RMSE**, **MAPE**  

---

## 📈 Results
The models produce forecasts close to actual consumption, with low error metrics:  
| Model  | MAE  | RMSE | MAPE (%) |
|--------|------|------|----------|
| SARIMA | 0.39 | 0.62 | 16.8     |
| ETS    | 0.39 | 0.61 | 16.7     |

Forecasts also include “noisy” versions to simulate realistic fluctuations in electricity consumption.

---

## 📊 Forecast Comparison: SARIMA vs ETS
Visual comparison of the forecasted electricity consumption for SARIMA and ETS models:  

![SARIMA vs ETS](Results/Charts/SARIMA%20VS%20ETS.png)

*Blue line:* SARIMA Forecast  
*Green line:* ETS Forecast  
*Black line:* Historical Consumption  


```bash
pip install pandas numpy matplotlib scikit-learn statsforecast
