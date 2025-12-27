import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import load_and_prepare_data
from evaluation import evaluate_forecast

# Load data
df = load_and_prepare_data("data/CPIAUCSL.csv")
series = df['inflation']

# Train-test split
train_size = int(len(series) * 0.8)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# ARIMA
arima_model = ARIMA(train, order=(1, 1, 1))
arima_fit = arima_model.fit()
forecast_arima = arima_fit.forecast(steps=len(test))

# SARIMA
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
sarima_fit = sarima_model.fit()
forecast_sarima = sarima_fit.forecast(steps=len(test))

# ETS
ets_model = ExponentialSmoothing(
    train, trend='add', seasonal='add', seasonal_periods=12
)
ets_fit = ets_model.fit()
forecast_ets = ets_fit.forecast(steps=len(test))

# Random Forest
def create_lag_features(series, lags=[1,3,6,12]):
    df_lag = pd.DataFrame(series)
    for lag in lags:
        df_lag[f'lag_{lag}'] = df_lag[series.name].shift(lag)
    return df_lag.dropna()

df_ml = create_lag_features(series)
X = df_ml.drop(columns=['inflation'])
y = df_ml['inflation']

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)
forecast_rf = rf_model.predict(X_test)

# Evaluation
models = {
    "ARIMA": forecast_arima,
    "SARIMA": forecast_sarima,
    "ETS": forecast_ets,
    "Random Forest": forecast_rf
}

print("Model Performance:")
for name, forecast in models.items():
    rmse, mae = evaluate_forecast(test, forecast)
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}")

# Visualization
plt.figure(figsize=(12,6))
plt.plot(test.index, test, label="Actual Inflation", color="black")
plt.plot(test.index, forecast_sarima, label="SARIMA Forecast")
plt.title("Inflation Forecast Using SARIMA")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.legend()
plt.tight_layout()
plt.savefig("results/forecast_plots.png")
plt.show()
