import pandas as pd
from loguru import logger
import xgboost as xgb
import numpy as np
from metrics import get_metrics
from utils import SpaceTimeSplits



if __name__=="__main__":
    train_data = pd.read_csv("../data/train_clean_v1.csv")
    val_data = pd.read_csv("../data/validation_v1.csv")

    label = "value"
    feats = [x for x in train_data.columns if x!=label]

    logger.debug(f"feats: {feats}")
    logger.debug(f"label: {label}")

    x, y = train_data[feats], train_data[label]
    x_test, y_test = val_data[feats], val_data[label]

    # Train XGB Model
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    xg_reg.fit(x,y)

    # Predict on test set
    pred_train = xg_reg.predict(x)
    logger.debug(f"y: {list(y)[0:5]}, pred_train: {pred_train}")
    metrics = get_metrics(np.exp(y), np.exp(pred_train))
    logger.info(f"train metrics: {metrics}")
    preds = xg_reg.predict(x_test)

    # Compute metrics
    # predicted = np.exp(preds)   # raise e to the results 
    # gt = np.exp(y_test)
    logger.debug(f"y_test: {list(y_test)[0:5]}, np.exp(preds): {np.exp(preds)}")
    metrics = get_metrics(y_test, np.exp(preds))
    logger.info(f"test metrics: {metrics}")

    
    # Compute metrics with current model
    # Possible resource if we want to predict with actual model https://stackoverflow.com/questions/59911156/how-to-load-a-model-saved-as-rds-file-from-python-and-make-predictions
    prev_actual, prev_pred = val_data["value"], val_data["Predicted"]
    orig_metrics = get_metrics(prev_actual, prev_pred)
    logger.info(f"orig_metrics: {orig_metrics}")
