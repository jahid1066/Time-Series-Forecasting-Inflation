# Time Series Forecasting of Inflation (CPI)

This project demonstrates **time series forecasting of monthly inflation rates** derived from the Consumer Price Index (CPI) using **statistical and machine learning methods**. It highlights quantitative, econometric, and machine learning modeling skills suitable for academic and industry data science applications.  

---

## Project Overview

Inflation forecasting is a critical task in economics and finance. This project uses historical CPI data to calculate monthly inflation rates and applies multiple forecasting models to predict future trends. The analysis includes:

- Data preprocessing and transformation
- Stationarity testing (ADF test)
- Statistical forecasting models (ARIMA, SARIMA, Exponential Smoothing)
- Machine learning approach (Random Forest regression)
- Model evaluation using RMSE and MAE
- Visualization of actual vs predicted inflation

---

## Dataset

- **Source:** [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/CPIAUCSL)
- **Indicator:** Consumer Price Index for All Urban Consumers (CPIAUCSL)
- **Frequency:** Monthly
- **Coverage:** 1947 – Present

Example:

| observation_date | CPIAUCSL | inflation |
|-----------------|----------|-----------|
| 1947-02-01      | 21.62    | 0.649654  |
| 1947-03-01      | 22.00    | 1.742364  |
| 1947-04-01      | 22.00    | 0.000000  |
| 1947-05-01      | 21.95    | -0.227531 |
| 1947-06-01      | 22.08    | 0.590508  |

---

## Methodology

1. **Data Preprocessing**
   - Imported CPI data and converted `observation_date` to datetime
   - Calculated monthly inflation using log differences
   - Removed missing values

2. **Stationarity Test**
   - Applied **Augmented Dickey-Fuller (ADF) test**
   - Series is stationary (ADF Statistic = -4.627, p-value = 0.0001)

3. **Statistical Models**
   - **ARIMA:** AutoRegressive Integrated Moving Average
   - **SARIMA:** Seasonal ARIMA for capturing seasonality
   - **ETS:** Exponential Smoothing (Additive trend + seasonality)

4. **Machine Learning Model**
   - **Random Forest Regression**
   - Created lag features to transform time series into supervised problem
   - Captures nonlinear patterns in inflation

5. **Evaluation Metrics**
   - **RMSE (Root Mean Squared Error)**
   - **MAE (Mean Absolute Error)**

---

## Results

| Model         | RMSE   | MAE   |
|---------------|--------|-------|
| ARIMA         | 0.268  | 0.193 |
| SARIMA        | 0.331  | 0.254 |
| ETS           | 0.280  | 0.202 |
| Random Forest | 0.246  | 0.183 |

- **Random Forest achieved the lowest RMSE and MAE**, effectively capturing non-linearities.  
- Statistical models remain competitive, especially for trend and seasonality interpretation.  

---


## How to Run

## 1. Clone the repository:

git clone https://github.com/jahid1066/Time-Series-Forecasting-Inflation.git

cd time-series-forecasting-inflation

## 2. Install dependencies:

pip install -r requirements.txt

## 3. Open the notebook:

- jupyter notebook notebooks/time_series_forecasting.ipynb

## 4. Run the notebook step by step to reproduce results

## Skills Demonstrated

- Time Series Analysis & Forecasting

- Time Series Analysis & Forecasting

- Econometrics (ARIMA, SARIMA, ETS)

- Machine Learning (Random Forest, Feature Engineering)

- Data Preprocessing & Transformation (Python, Pandas, Numpy)

- Model Evaluation & Metrics (RMSE, MAE)

- Data Visualization (Matplotlib)

- GitHub Workflow & Project Organizatio

## Future Work

- Incorporate additional macroeconomic indicators (GDP, unemployment)

- Implement advanced ML models (XGBoost, LSTM, Prophet)

- Perform cross-validation for hyperparameter tuning

- Automate forecasting pipeline with cloud deployment

## References

- FRED CPIAUCSL Data

- Hyndman, R.J., Athanasopoulos, G. Forecasting: Principles and Practice

- James, G., Witten, D., Hastie, T., Tibshirani, R. An Introduction to Statistical Learning

## Author

Md Jahidul Islam