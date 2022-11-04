TO_LOAD_CKPT = False     # Set to False to train model; True to load a model checkpoint
TRAINVAL_DATA_PATH = "../data/train_raw.csv"
TRAIN_SPLIT_DATA_PATH = "../data/train_rawdata_split.csv"
VAL_SPLIT_DATA_PATH = "../data/val_rawdata_split.csv"
TEST_DATA_PATH = "../data/validation_v1.csv"
CHECKPOINT_PATH = "../checkpoints"
# LABEL = "value"
LABEL = "value"
FEATS_TO_DISCARD = [
    "Unnamed: 0",   # just the row number
    "COMID",
    
    # Grouping variables
    ".partitions",
    "julian",   # kind of represented by "date_unity" field already
    "space_group",
    "lat_group",
    "long_group",
    "time_group",
    "mag",      # related to value we're trying to predict
    
    "SiteID",
    "endtime",
    "date",
    "date_utc",
    "time",
    "landsat_id",
    "hexcolor",     # 753 possible values, will consider encoding later
    "uniqueID",
    "n",    # unknown
    "FTYPE_lake",   # duplicate of "ftype_lake"
    "ftype_lake",   # seems noisy (can add back later)
    "",
    "HWTYPE",   # no values ("A", "NA")

    # Empty columns
    "HWNodeSqKM",   # empty column
    "TOT_EVI_AMJ_2012", # empty column

    # Single valued cols
    "dswe",     # all are 1
    "source",   #all are "WQP"
    "characteristicName",   # all are TSS
    "parameter",   # all are TSS
    "harmonized_unit",  # same values for all rows (mg/l)
    "date_only",    # all are False
    "TZID", #timezone id (all are UTC)

    "hillshadow",
    "NS_NR",
    "StartFlag",
    "VPUIn",
    "MAXELEVRAW",
    "Enabled",

    # Single valued (all zeros)
    "TOT_ECOL3_30", # single value ("0")
    "TOT_ECOL3_31",     # single value
    "TOT_S1720",
    "TOT_ECOL3_34",
    "ACC_ECOL3_30",
    "ACC_ECOL3_31",
    
    "ACC_ECOL3_34", # all are 0

    # Too many values to be encoded (can be put back later)
    "analytical_method",  # 58 unique vals
    "GNIS_ID",    # 1149 unique vals
    "GNIS_NAME",  # 1010 unique vals
    "RPUID",  # 55 unique vals
    "VPUID",  # 22 unique vals
    "gnis_name_lake", # 1212 unique vals

    "",
    " ",
]

DATE_FEATS = ["date_unity", "FDATE"]
CATEGORICAL_FEATS = [
    "SiteID",
    "sat",
    "type",
    "endtime",
    "date",
    "TZID",
    "date_utc",
    "time",
    "landsat_id",
    "characteristicName",
    "analytical_method",
    "harmonized_unit",
    "source",
    "hexcolor",
    "parameter",
    "RESOLUTION",
    "GNIS_ID",
    "GNIS_NAME",
    "FLOWDIR",
    "FTYPE",
    "WBAreaType",
    "HWTYPE",
    "RPUID",
    "VPUID",
    "gnis_name_lake",
    "ftype_lake",
    "lat_group",
    "long_group",
    "space_group",
    "time_group",
]
FEATS_TO_ENCODE = CATEGORICAL_FEATS
TO_REMOVE_INC_COLS = False
TO_REMOVE_INC_ROWS = False
# feats = [x for x in train_data.columns if x!=label]
# NARROWED_FEATS = ['nir', 'R.BS', 'swir2', 'N_R', 'BG', 'swir1', 'RG', 'bright', 'GR'] # from John
# NARROWED_FEATS = ['GN2', 'G.BN', 'blue', 'R.NS', 'B.RG', 'B.GS', 'ndvi', 'Lake', 'G.BS', 'Estuary', 'path', 'bright', 'rn', 'G.NS', 'dw', 'B.RS', 'timediff', 'hue', 'R.BN', 'BR_G', 'R.BS', 'BR', 'GR', 'BG', 'RG', 'Stream', 'Facility', 'red', 'clouds', 'lat', 'long', 'row', 'swir1_sd', 'azimuth', 'zenith', 'area.km2', 'pwater', 'nir_raw', 'RS', 'N.BS', 'NR', 'R.GN', 'N.GB', 'N.RS', 'N.GS', 'NG', 'B.RN', 'GS', 'swir2', 'swir1_raw', 'red_raw', 'green_raw', 'bright_tot', 'LT05', 'blue_raw', 'swir2_raw', 'G.RN', 'N.RG', 'N.RB', 'B.GN', 'ndssi', 'R.GB', 'G.RB', 'gn.gn', 'G.BR']
# FEATS_TO_ENCODE = ["sat", "type"]

GROUP_NAMES = ["time_group", "space_group"]
TO_NORM_FEATS = True # set to True to normalize features
TO_NORM_LABELS = False # set to True to normalize labels
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

# For MLP model
NUM_EPOCHS = 100
EVAL_EPOCH_EVERY = 10

# For Linear Model
LR_TYPE = 'lr'
# Choices: 'lr', 'ridgecv', 'sgdreg', 'elasticCV"


TARGET_METRICS = {
    'rmse': 30.874488274822923,
    'mae': 10.108542659950823,
    'mape': 0.5554461311204274,
    'r2': 0.8229691722882178,
    'bias': 5.260115649343916,
    'pct_bias': -0.2124630453389412,
    'smape': 0.4586837552555118
}