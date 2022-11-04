from typing import List
import numpy as np
import random
from loguru import logger
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, RidgeCV, SGDRegressor, ElasticNetCV

# Import local modules
from metrics import get_metrics
from utils import SpaceTimeSplits, normalize_test_df_feat, normalize_train_df_feat
from constants import (
    TO_LOAD_CKPT,
    TRAINVAL_DATA_PATH,
    CHECKPOINT_PATH,
    LABEL,
    NARROWED_FEATS,
    GROUP_NAMES,
    TO_NORM_FEATS,
    TO_NORM_LABELS,
    IS_PRED_EXP,
    GRIDSEARCH_CV_SCORING,
    CV_FOLDS,
    NUM_EPOCHS,
    EVAL_EPOCH_EVERY,
    LR_TYPE,
    FEATS_TO_ENCODE,
)

pd.options.mode.chained_assignment = None   # to disable chained assignment warning (by default, this is "warn")

def encode_df(train_df, test_df):
    encoded_feats = []
    for col in FEATS_TO_ENCODE:
        one_hot = pd.get_dummies(train_df[col])
        encoded_feats += list(one_hot.columns)
        train_df = train_df.drop(col,axis = 1)
        train_df = train_df.join(one_hot)

        one_hot = pd.get_dummies(test_df[col])
        encoded_feats += list(one_hot.columns)
        test_df = test_df.drop(col,axis = 1)
        test_df = test_df.join(one_hot)

    train_df=train_df.reset_index()
    test_df=test_df.reset_index()

    return train_df, test_df

def prep_feats(
    feats: List[str],
    train_df: pd.DataFrame,
    X: pd.DataFrame,
    y,
    x_test: pd.DataFrame,
    y_test,
):

    label_mu, label_sigma = None, None
    if TO_NORM_FEATS:
        logger.debug(f"Normalizing features, TO_NORM_FEATS: {TO_NORM_FEATS}")
        norm_params = {x:None for x in feats}
        for col_name in feats:
            # Normalize train features
            col_val, mu, sigma = normalize_train_df_feat(X, col_name)
            X.loc[:, (col_name)] = col_val
            norm_params[col_name] = [mu, sigma]

            # Normalize test features
            col_val = normalize_test_df_feat(x_test, col_name, mu, sigma)
            x_test.loc[:, (col_name)] = col_val
    if TO_NORM_LABELS:
        logger.debug("Normalizing labels")
        col_val, label_mu, label_sigma = normalize_train_df_feat(train_df, LABEL)
        y = col_val.values.astype('float32')

    return X, y, x_test, y_test, label_mu, label_sigma

if __name__ == "__main__":
    train_df = pd.read_csv(TRAINVAL_DATA_PATH)
    test_data = pd.read_csv("../data/validation_v1.csv")

    train_df, test_data = encode_df(train_df, test_data)

    x_test, y_test = test_data[NARROWED_FEATS], test_data[LABEL]

    # store the inputs and outputs
    X = train_df[NARROWED_FEATS].astype('float32')
    y = train_df[LABEL].values.astype('float32')
    logger.debug(f"Features: {NARROWED_FEATS}")
    logger.debug(f"Label: {LABEL}")

    # Preprocess features
    X, y, x_test, y_test, label_mu, label_sigma = prep_feats(
        NARROWED_FEATS, train_df, X, y, x_test, y_test
    )
    groups = train_df[GROUP_NAMES]

    # Define Cross Validation method
    st_cv = SpaceTimeSplits(n_splits=CV_FOLDS)
    custom_splitter = st_cv.split(
        X,
        y,
        groups=groups,
    )
    cv_iter = []
    for t_idx, v_idx in custom_splitter:
        cv_iter.append( (t_idx, v_idx) )
        logger.debug(f"CV Split: t_idx: {len(t_idx)}, v_idx: {len(v_idx)}")
        for i in v_idx: # Just to make sure there are no overlaps
            if i in t_idx:
                logger.error(f"{i} in t_idx!")

    # Fit Model
    logger.debug("Fitting LR model")
    if LR_TYPE == "lr":
        lm = LinearRegression()
        model = lm.fit(X,y)
    elif LR_TYPE == "ridgecv":
        # list of alphas to check: 100 values from 0 to 5 with
        r_alphas = np.logspace(0, 5, 10)
        lm = RidgeCV(
            alphas=r_alphas,
            scoring=GRIDSEARCH_CV_SCORING,
            cv=cv_iter,
        )
        y = y.reshape(-1,1)
        model = lm.fit(X.values,y)
        logger.info(f"lm.alpha_: {lm.alpha_}, intercept: {lm.intercept_}, coef: {lm.coef_}")
    elif LR_TYPE == "sgdreg":
        lm = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            l1_ratio=0.15,
            eta0=0.01,
        )
        model = lm.fit(X,y)
    elif LR_TYPE == "elasticCV":
        lm = ElasticNetCV(
            cv=cv_iter,
        )
        model = lm.fit(X,y)
    else:
        raise NotImplementedError

    # Get predictions and metrics
    test_preds = model.predict(x_test).reshape(-1,)
    lr_pred = test_preds
    if TO_NORM_LABELS:
        # reverse normalization
        test_pred = (test_preds*label_sigma) + label_mu
    if IS_PRED_EXP:
        test_metrics = get_metrics(y_test, test_preds)
    else:
        test_metrics = get_metrics(y_test,  np.round(np.exp(test_preds),2))
        lr_pred = np.round(np.exp(test_preds),2)
    print(f"test_metrics: {test_metrics}")

    # results = pd.DataFrame(
    #     np.vstack([lr_pred, y_test]).T,
    #     columns=["LR","GT"]
    # )
    # results.to_csv("../raw_lr_results.csv")