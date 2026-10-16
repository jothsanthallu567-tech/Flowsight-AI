import numpy as np

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - (np.mean(y_true))) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
