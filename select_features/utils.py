import random
import numpy as np
import pandas as pd
from loguru import logger

def preprocess_feats(
    train_df,
    test_df,
    label,
    feats_to_encode=[],
    feats_to_discard=[],
    group_names=[],
    to_remove_incomplete_cols=True,
    to_remove_rows_w_invalid_vals=True,
    to_norm_feats=True,
    to_norm_labels=False,
):
    train_data = train_df.copy()
    test_data = test_df.copy()

    # Remove uninformative columns
    logger.debug(f"Removing the ff columns: {feats_to_discard}")
    for col in feats_to_discard:
        if col in train_data.columns:
            train_data = train_data.drop(col, axis=1)
        
    # One hot encode features
    encoded_feats = []
    for col in feats_to_encode:
        if col in group_names:
            continue
        if (col not in train_data.columns) or (col not in test_data.columns):
            continue
        if col in feats_to_discard:
            continue
        one_hot = pd.get_dummies(train_data[col], prefix=f"{col}")
        encoded_feats += list(one_hot.columns)
        train_data = train_data.drop(col,axis = 1)
        train_data = train_data.join(one_hot)

        one_hot = pd.get_dummies(test_data[col], prefix=f"{col}")
        encoded_feats += list(one_hot.columns)
        test_data = test_data.drop(col,axis = 1)
        test_data = test_data.join(one_hot)
    logger.debug(f"Encoded {len(encoded_feats)} features: {encoded_feats}")

    if to_remove_incomplete_cols:
        logger.debug(f"Removing columns with missing values")
        ctr = 0
        tot_ctr = 0
        incomplete_cols = []
        for col in train_data.columns:
            if col==label or col in feats_to_discard:
                continue
            tot_ctr += 1
            num_missing = np.sum(train_data[col].isnull())
            if num_missing>0:
                ctr += 1
                # print(col, num_missing)
                incomplete_cols.append(col)
        logger.debug(f"Found {ctr} out of {tot_ctr} features with missing values")

        # Remove cols with missing values
        for col in incomplete_cols:
            if col in train_data.columns:
                train_data = train_data.drop(col, axis=1)
            
    
    # Remove rows with nan and inf
    if to_remove_rows_w_invalid_vals:
        for col in train_data.columns:
            num_na = train_data[col].isna().sum()
            if num_na > 0:
                logger.debug(f"{col} has {num_na} NA values")
            num_null = train_data[col].isnull().sum()
            if num_null > 0:
                logger.debug(f"{col} has {num_null} null values")

        # Replace inf values with nan
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.dropna()
        logger.debug(f"Removed rows with invalid values {train_data.shape}")

    if to_norm_feats:
        logger.debug(f"Normalizing features")
        norm_params = {}
        for col in train_data.columns:
            if col==label:  # skip label
                continue
            if (col in feats_to_encode) or (col in encoded_feats):
                continue    # don't normalize one hot encoded features
            if col in group_names:
                continue    # don't normalize groups
            if col in feats_to_encode:
                continue
            if col in train_data.columns and col in test_data.columns:
                # Normalize train features
                col_val, mu, sigma = normalize_train_df_feat(train_data, col)
                train_data.loc[:, (col)] = col_val
                norm_params[col] = [mu, sigma]

                # Normalize test features with train set's mu and sigma
                col_val = normalize_test_df_feat(test_data, col, mu, sigma)
                test_data.loc[:, (col)] = col_val
        # logger.debug(f"norm_params: {norm_params}")
    label_mu, label_sigma = None, None
    if to_norm_labels:
        logger.debug(f"Normalizing label")
        col_val, label_mu, label_sigma = normalize_train_df_feat(train_data, label)
        train_data[label] = col_val

    train_data = train_data.reset_index()
    return train_data, test_data, label_mu, label_sigma

def normalize_train_df_feat(df, col_name):
    mu = np.mean(df[col_name])
    sigma = np.var(df[col_name])**0.5
    return (df[col_name]-mu)/sigma, mu, sigma

def normalize_test_df_feat(df, col_name, mu, sigma):
    return (df[col_name]-mu)/sigma

class SpaceTimeSplits():
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups:pd.DataFrame=None):
        """
        groups = [[long_group, time_group]] values for each row
        Implementation based on https://github.com/HannaMeyer/CAST/blob/master/R/CreateSpacetimeFolds.R
        """
        unique_space_groups = np.unique(groups["space_group"])
        unique_time_groups = np.unique(groups["time_group"])
        random.shuffle(unique_space_groups)
        random.shuffle(unique_time_groups)

        for i in range(self.n_splits):
            cur_space_group = unique_space_groups[i]
            cur_time_group = unique_time_groups[i]
            
            # Get indices of elements with same space group and same time group (val)
            val_idx = groups.index[
                (groups["space_group"]==cur_space_group) & 
                (groups["time_group"]==cur_time_group)
            ]

            # Get indices of elements with diff space group and diff time group (train)
            train_idx = groups.index[
                (groups["space_group"]!=cur_space_group) & 
                (groups["time_group"]!=cur_time_group)
            ]
        
            yield list(train_idx), list(val_idx)



def adjusted_rmse(y_true, y_pred, **kwargs):
    # print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    if IS_PRED_EXP:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        rmse = np.sqrt(mean_squared_error(np.exp(y_true), np.exp(y_pred)))
    # print(f"rmse: {rmse}")
    return rmse

def adjusted_mae(y_true, y_pred, **kwargs):
    # print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    if IS_PRED_EXP:
        mae = mean_absolute_error(y_true, y_pred)
    else:
        mae = mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    # print(f"mae: {mae}")
    return mae

def adjusted_mape(y_true, y_pred, **kwargs):
    # print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    if IS_PRED_EXP:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    else:
        mape = mean_absolute_percentage_error(np.exp(y_true), np.exp(y_pred))
    # print(f"mape: {mape}")
    return mape


def adjusted_r2(y_true, y_pred, **kwargs):
    # print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")
    if IS_PRED_EXP:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = r2_score(np.exp(y_true), np.exp(y_pred))
    # print(f"r2: {r2}")
    return r2

if __name__=="__main__":
    # Test the function
    st_cv = SpaceTimeSplits(3)
    train_data = pd.read_csv("../data/train_raw.csv")
    # val_data = pd.read_csv("../data/validation_v1.csv")

    label = "value"
    # feats = [x for x in train_data.columns if x!=label]
    feats = ['nir', 'R.BS', 'swir2', 'N_R', 'BG', 'swir1', 'RG', 'bright', 'GR']
    group_names = ["time_group", "space_group"]

    logger.debug(f"feats: {feats}")
    logger.debug(f"label: {label}")

    x, y, groups = train_data[feats], train_data[label], train_data[group_names]
    custom_splitter = st_cv.split(
        x,
        y,
        groups=groups
    )
    iter = 0
    for train_idx, val_idx in custom_splitter:
        if iter>=10:
            break
        iter += 1
        logger.debug(f"Iteration {iter}")
        logger.debug(f"train_idx:{len(train_idx)}; val_idx: {len(val_idx)}")
        for idx in train_idx:
            if idx in val_idx:
                logger.error(f"{idx} is duplicate!")