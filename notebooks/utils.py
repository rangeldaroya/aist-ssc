import random
import numpy as np
import pandas as pd
from loguru import logger

class SpaceTimeSplits():
    def __init__(self, n_splits=3):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups:pd.DataFrame=None):
        """
        groups = [[long_group, time_group]] values for each row
        Implementation based on https://github.com/HannaMeyer/CAST/blob/master/R/CreateSpacetimeFolds.R
        and https://rdrr.io/cran/CAST/man/CreateSpacetimeFolds.html 
        """
        # logger.debug(f"Entering function")
        unique_space_groups = np.unique(groups["space_group"])
        unique_time_groups = np.unique(groups["time_group"])
        # while True:
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
            # logger.debug(f"Splitting data i={i}, train_idx: {len(train_idx)}, val_idx: {len(val_idx)}")
        
            yield list(train_idx), list(val_idx)

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