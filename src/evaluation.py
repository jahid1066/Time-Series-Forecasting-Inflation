import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    return rmse, mae
