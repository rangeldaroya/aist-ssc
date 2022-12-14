import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error,
    r2_score,
)
from loguru import logger

def get_smape(gt, pred):
    num = np.abs(pred-gt)
    denom = (np.abs(gt) + np.abs(pred)) / 2
    return np.sum(num/denom)/len(gt)

def get_pct_bias(gt, pred):
    # Based on https://rdrr.io/cran/Metrics/man/percent_bias.html
    diff = (gt-pred)/np.abs(gt)
    return np.mean(diff)

def get_bias(gt, pred):
    # Based on https://rdrr.io/cran/Metrics/man/bias.html
    return np.mean(gt - pred)

def get_metrics(gt, pred):
    # Compute metrics
    metrics = {
        "rmse": np.sqrt(mean_squared_error(gt, pred)),
        "mae": mean_absolute_error(gt, pred),
        "mape": mean_absolute_percentage_error(gt, pred),
        "r2": r2_score(gt, pred),
        "bias": get_bias(gt, pred),
        "pct_bias": get_pct_bias(gt, pred),
        "smape": get_smape(gt, pred),
    }
    return metrics