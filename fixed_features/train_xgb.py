import pandas as pd
from constants import TARGET_METRICS
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

from utils import SpaceTimeSplits, preprocess_feats
from metrics import get_metrics

pd.options.mode.chained_assignment = None

FEATS_TO_ENCODE = []
FEATS_TO_DISCARD = []
TO_NORM_FEATS = True # set to True to normalize features
TO_NORM_LABELS = False # set to True to normalize labels
TO_REMOVE_INC_COLS = True
TO_REMOVE_INC_ROWS = False
IS_PRED_EXP = False  # Set to True to directly predict TSS as np.exp(value). otherwise, estimate the ln of the actual TSS value and just modify for measuring metrics
GRIDSEARCH_CV_SCORING = 'neg_root_mean_squared_error'
"""Other options for grid cv scoring:
 'neg_mean_absolute_error',
 'neg_mean_absolute_percentage_error',
 'neg_mean_gamma_deviance',
 'neg_mean_poisson_deviance',
 'neg_mean_squared_error',
 'neg_mean_squared_log_error',
 'neg_median_absolute_error',
 'neg_root_mean_squared_error',
 'r2',
 """

CV_FOLDS = 3
train_data = pd.read_csv("../data/train_raw.csv")
val_data = pd.read_csv("../data/validation_v1.csv")
label = "value"
feats = ['nir', 'R.BS', 'swir2', 'N_R', 'BG', 'swir1', 'RG', 'bright', 'GR']
group_names = ["time_group", "space_group"]
train_data = train_data[feats+group_names+[label]]
val_data = val_data[feats+[label]]

train_data, test_data, label_mu, label_sigma = preprocess_feats(
    train_data,
    val_data,
    label,
    feats_to_encode=FEATS_TO_ENCODE,
    feats_to_discard=FEATS_TO_DISCARD,
    group_names=group_names,
    to_remove_incomplete_cols=TO_REMOVE_INC_COLS,
    to_remove_rows_w_invalid_vals=TO_REMOVE_INC_ROWS,
    to_norm_feats=TO_NORM_FEATS,
    to_norm_labels=TO_NORM_LABELS,
)

x, y, groups = train_data[feats], train_data[label], train_data[group_names]
x_test, y_test = test_data[feats], test_data[label]


if IS_PRED_EXP:
    y = np.exp(y)


params = {
 'alpha': 0.01,
 'lambda': 1,
 'eta': 0.3,    # learning_rate
 'n_estimators': 50, # num rounds
 'max_depth': 5,
 'objective':'reg:squarederror',
 "booster": "gbtree",
}

xgb_reg = xgb.XGBRegressor(**params)
xgb_reg.fit(x,y)
train_pred = xgb_reg.predict(x)


if IS_PRED_EXP:
    train_metrics = get_metrics(y, train_pred)
else:
    train_metrics = get_metrics(np.exp(y), np.exp(train_pred))

test_pred = xgb_reg.predict(x_test)
if TO_NORM_LABELS:
    # reverse normalization
    test_pred = (test_pred*label_sigma) + label_mu
if IS_PRED_EXP:
    test_metrics = get_metrics(y_test, test_pred)
else:
    test_metrics = get_metrics(y_test,  np.round(np.exp(test_pred),2))

# Display metrics
train_metrics["source"] = "train set"
test_metrics["source"] = "test set"
TARGET_METRICS["source"] = "test set target"
ds = [train_metrics, test_metrics, TARGET_METRICS]
print(pd.DataFrame(ds))
