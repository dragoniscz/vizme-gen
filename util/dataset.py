from scipy.cluster.hierarchy import *
import scipy.cluster.hierarchy as spc
import pandas as pd


def get_feature_ordering_order_based_on_correlation(df):
    df_corr = df.drop(['id'], axis=1, errors="ignore").corr()
    pdist = spc.distance.pdist(df_corr)
    Z = ward(pdist)
    features_order = leaves_list(Z)
    return pd.Series(features_order)
