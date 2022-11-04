import pandas as pd
from config_mlp import (
    GROUP_NAMES,
    FEATS,
    LABEL,
    CATEGORICAL_FEATS,
    FEATS_TO_ENCODE,
    DATE_FEATS,
    TARGET_METRICS,
    FEATS_TO_DISCARD,
)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics._scorer import make_scorer
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.neural_network import MLPRegressor

from utils import SpaceTimeSplits, preprocess_feats
from metrics import get_metrics

pd.options.mode.chained_assignment = None

TO_NORM_FEATS = True # set to True to normalize features
TO_NORM_LABELS = False # set to True to normalize labels
TO_REMOVE_INC_COLS = True
TO_REMOVE_INC_ROWS = False
IS_PRED_EXP = False  # Set to True to directly predict TSS as np.exp(value). otherwise, estimate the ln of the actual TSS value and just modify for measuring metrics


CV_FOLDS = 3
train_data = pd.read_csv("../data/train_raw.csv")
train_data_raw = train_data.copy()
val_data = pd.read_csv("../data/test_raw.csv")

if "Unnamed: 0" in train_data.columns:
    train_data = train_data.drop("Unnamed: 0", axis=1)
# Replace missing values with mean (using only training data)
for col in train_data.columns:
    if col in DATE_FEATS:
        continue
    if col in CATEGORICAL_FEATS:
        continue
    if col in FEATS_TO_DISCARD:
        continue
    col_mean = train_data[col].mean()
    train_data[col] = train_data[col].replace(np.nan, col_mean)
    val_data[col] = val_data[col].replace(np.nan, col_mean)


train_data, test_data, label_mu, label_sigma = preprocess_feats(
    train_data,
    val_data,
    label=LABEL,
    feats_to_encode=FEATS_TO_ENCODE,
    feats_to_discard=FEATS_TO_DISCARD+DATE_FEATS,
    group_names=GROUP_NAMES,
    to_remove_incomplete_cols=TO_REMOVE_INC_COLS,
    to_remove_rows_w_invalid_vals=TO_REMOVE_INC_ROWS,
    to_norm_feats=TO_NORM_FEATS,
    to_norm_labels=TO_NORM_LABELS,
)

x, y, groups = train_data[FEATS], train_data[LABEL], train_data_raw[GROUP_NAMES]
x_test, y_test = test_data[FEATS], test_data[LABEL]
y_test = np.exp(y_test) # need to np.exp since getting from raw values

if "index" in train_data.columns:
    train_data = train_data.drop("index", axis=1)
if "index" in test_data.columns:
    test_data = test_data.drop("index", axis=1)

logger.debug(f"Train data shapes: x: {x.shape}, y: {y.shape}")
logger.debug(f"Test data shapes: x: {x_test.shape}, y: {y_test.shape}")

if IS_PRED_EXP:
    y = np.exp(y)


params = {
    'batch_size': 1024,
    'learning_rate': 'adaptive',
    'max_iter': 50,
    'momentum': 0.8669252871209164
}

logger.debug(f"Fitting model...")
mlp_reg = MLPRegressor(
    **params,
    activation="tanh",
    solver="sgd",
    early_stopping=True,
)
mlp_reg.fit(x.values,y.values)
train_pred = mlp_reg.predict(x.values)


if IS_PRED_EXP:
    train_metrics = get_metrics(y, train_pred)
else:
    train_metrics = get_metrics(np.exp(y), np.exp(train_pred))

test_pred = mlp_reg.predict(x_test.values)
if TO_NORM_LABELS:
    # reverse normalization
    test_pred = (test_pred*label_sigma) + label_mu
if IS_PRED_EXP:
    test_metrics = get_metrics(y_test, test_pred)
else:
    test_metrics = get_metrics(y_test,  np.round(np.exp(test_pred),2))
    test_pred = np.round(np.exp(test_pred),2)

# Display metrics
train_metrics["source"] = "train set"
test_metrics["source"] = "test set"
TARGET_METRICS["source"] = "test set target"
ds = [train_metrics, test_metrics, TARGET_METRICS]
print(pd.DataFrame(ds))


pd.DataFrame(list(zip(test_pred, y_test.values)),
               columns =["MLP", "GT"]).to_csv("optuna_MLP_results.csv", index=False)