LABEL = "value"
FEATS = ['elevation', 'path', 'swir1_sd', 'lat', 'red_raw',# 'rn',
       'green_raw', 'blue_raw', 'nir_raw', 'swir1_raw', 'swir2_raw', 'RN', 'GS',
       'RS', 'NS', 'R.GN', 'R.GB', 'R.GS', 'R.BN', 'R.BS', 'R.NS', 'G.BN',
       'G.BS', 'N.GB', 'N.BS', 'GR2', 'GN2', 'N_S', 'ndwi', 'bright']   # 32.413560  10.865098  0.562409  0.804880  4.643504 -0.223400  0.455521
# FEATS = ['elevation', 'path', 'swir1_sd', 'lat', 'red_raw',# 'rn',
#        'green_raw', 'blue_raw', 'nir_raw', 'swir1_raw', 'swir2_raw',
#        'red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'RG', 'RN', 'GS',
#        'RS', 'NS', 'R.GN', 'R.GB', 'R.GS', 'R.BN', 'R.BS', 'R.NS', 'G.BN',
#        'G.BS', 'N.GB', 'N.BS', 'GR2', 'GN2', 'N_S', 'ndwi', 'bright'] # 30.415173  10.793213  0.563498  0.828197  4.548310 -0.225414  0.454611
# FEATS = ['elevation', 'path', 'swir1_sd', 'lat', 'red_raw',# 'rn',
#        'green_raw', 'blue_raw', 'nir_raw', 'swir1_raw', 'swir2_raw',
#        'red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'RG', 'RN', 'GS',
#        'RS', 'NS', 'R.GN', 'R.GB', 'R.GS', 'R.BN', 'R.BS', 'R.NS', 'G.BN',
#        'G.BS', 'N.GB', 'N.BS', 'GR2', 'GN2', 'N_S', 'ndwi', 'bright',
#        'bright_tot', 'dw', 'LENGTHKM', 'REACHCODE', 'WBAREACOMI',
#        'Shape_Length', 'StreamLeve', 'StreamOrde', 'StreamCalc',
#        'FromNode', 'ToNode', 'Hydroseq', 'LevelPathI', 'Pathlength',
#        'TerminalPa', 'ArbolateSu', 'DnLevel', 'UpLevelPat', 'UpHydroseq',
#        'DnLevelPat', 'DnHydroseq', 'RtnDiv', 'TotDASqKM', 'DivDASqKM',
#        'MINELEVRAW', 'MAXELEVSMO', 'MINELEVSMO', 'SLOPELENKM', 'VA_MA',
#        'VC_MA', 'VE_MA', 'RAreaHLoad', 'gnis_id_lake', 'reachcode_lake',
#        'ONOFFNET_lake', 'TOT_BASIN_AREA', 'TOT_BASIN_SLOPE',
#        'TOT_ELEV_MEAN', 'TOT_ELEV_MIN', 'TOT_ELEV_MAX',
#        'TOT_STREAM_SLOPE', 'TOT_STREAM_LENGTH', 'TOT_BUSHREED5',
#        'ACC_ECOL3_7', 'ACC_ECOL3_13', 'ACC_ECOL3_18', 'ACC_ECOL3_20',
#        'ACC_ECOL3_21', 'ACC_ECOL3_22', 'ACC_ECOL3_25', 'ACC_ECOL3_27',
#        'ACC_ECOL3_33', 'ACC_ECOL3_46', 'ACC_ECOL3_47', 'ACC_ECOL3_48',
#        'ACC_ECOL3_55', 'TOT_ECOL3_7', 'TOT_ECOL3_18', 'TOT_ECOL3_19',
#        'TOT_ECOL3_20', 'TOT_ECOL3_21', 'TOT_ECOL3_22', 'TOT_ECOL3_25',
#        'TOT_ECOL3_27', 'TOT_ECOL3_33', 'TOT_ECOL3_46', 'TOT_ECOL3_47',
#        'TOT_ECOL3_48', 'TOT_ECOL3_55', 'ACC_NDAMS2013', 'ACC_MAJOR2013',
#        'TOT_NDAMS2013', 'TOT_MAJOR2013', 'TOT_NLCD11_12', 'TOT_NLCD11_31',
#        'TOT_NLCD11_52', 'TOT_NLCD11_71', 'TOT_NLCD11_82', 'ACC_RDX',
#        'TOT_RDX', 'TOT_S1500', 'sinuosity', 'TOT_HGB', 'TOT_KFACT',
#        'TOT_KFACT_UP', 'TOT_NO200AVE', 'TOT_AWCAVE', 'TOT_WTDEP',
#        'TOT_SILTAVE', 'TOT_CLAYAVE', 'type_Stream', 'RESOLUTION_Medium',
#        'FLOWDIR_With Digitized', 'WBAreaType_StreamRiver']

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
# FEATS_TO_ENCODE = ["sat", "type"]
FEATS_TO_ENCODE = CATEGORICAL_FEATS
TO_NORM_FEATS = True # set to True to normalize features
TO_NORM_LABELS = False # set to True to normalize labels
TO_REMOVE_INC_COLS = False
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
GROUP_NAMES = ["time_group", "space_group"]
TARGET_METRICS = {
    'rmse': 30.874488274822923,
    'mae': 10.108542659950823,
    'mape': 0.5554461311204274,
    'r2': 0.8229691722882178,
    'bias': 5.260115649343916,
    'pct_bias': -0.2124630453389412,
    'smape': 0.4586837552555118
}

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
    "rn",   # Identifier
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