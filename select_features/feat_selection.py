from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import r_regression, SelectPercentile, SelectKBest
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_pca_feats(x, x_test, y):
    pca = PCA(n_components = 30)
    
    X_train_pca = pca.fit_transform(x)
    X_test_pca = pca.transform(x_test)
  
    # explained_variance = pca.explained_variance_ratio_
    # explained_variance
    return X_train_pca, X_test_pca

def get_f_feats(x, x_test, y, sorted_fvals, FVAL_THRESH = 500):
    # f_corr = f_regression(x, y)
    # fvals = {}
    # for idx, fval in enumerate(f_corr[0]):
    #     fvals[x.columns[idx]] = fval

    # sorted_fvals = {k: v for k, v in sorted(fvals.items(), key=lambda item: item[1])}
    f_feats = [k for k,v in sorted_fvals.items() if (v>FVAL_THRESH and k in x.columns)]

    return x[f_feats].values, x_test[f_feats].values


def get_mi_feats(x, x_test, y, sorted_mi_vals, MI_THRESH = 0.2):
    # mi_reg_corr = mutual_info_regression(x, y)
    # mi_vals = {}
    # for idx, mi_val in enumerate(mi_reg_corr):
    #     mi_vals[x.columns[idx]] = mi_val
    
    # sorted_mi_vals = {k: v for k, v in sorted(mi_vals.items(), key=lambda item: item[1]) if v>MI_THRESH}
    mi_feats = [k for k,v in sorted_mi_vals.items() if (v>MI_THRESH and k in x.columns)]

    return x[mi_feats].values, x_test[mi_feats].values


def get_feat_percentile(x, x_test, y, percentile=30):

    filt = SelectPercentile(r_regression, percentile=percentile)
    X_selection = filt.fit_transform(x, y)
    pearson_feats_perc = filt.get_feature_names_out()

    return x[pearson_feats_perc].values, x_test[pearson_feats_perc].values


def get_feat_num(x, x_test, y, k=30):
    filt = SelectKBest(r_regression, k=k)
    X_selection = filt.fit_transform(x, y)
    pearson_feats_num = filt.get_feature_names_out()

    return x[pearson_feats_num].values, x_test[pearson_feats_num].values


def get_vif_feats(x, x_test, y, vif_scores, VIF_THRESH = 10):
    # Variance inflation factor measures how much the behavior (variance) of an independent variable is influenced, or inflated, 
    # by its interaction/correlation with the other independent variables. 
    # Variance inflation factors allow a quick measure of how much a variable is contributing to the standard error in the regression.
    # Values of more than 4 or 5 are sometimes regarded as being moderate to high, with values of 10 or more being regarded as very high.
    # vif_scores = {x.columns[feature]: variance_inflation_factor(x.values, feature) for feature in range(len(x.columns))}

    vif_feats = []
    for col, vif in vif_scores.items():
        if vif>VIF_THRESH:
            continue
        if col not in x.columns:
            continue
        vif_feats.append(col)
    
    return x[vif_feats].values, x_test[vif_feats].values