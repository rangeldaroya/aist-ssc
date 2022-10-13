import random
import numpy as np
import pandas as pd

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