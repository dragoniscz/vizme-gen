from scipy.cluster.hierarchy import *
import scipy.cluster.hierarchy as spc
import pandas as pd
import numpy as np
from loguru import logger


def get_feature_ordering_order_based_on_correlation(df):
    """
    Computes the correlation matrix on the feature values and then reorder the features to get the correlated f
    feature closer. Use the hierarchical clustering on the correlation matrix to get the feature order.
    :param df: Data to compute the feature, ordering from
    :return:
    """
    df_corr = df.drop(['id'], axis=1, errors="ignore").corr()
    pdist = spc.distance.pdist(df_corr)
    Z = ward(pdist)
    features_order = leaves_list(Z)
    return pd.Series(features_order)


#
def get_samples_for_labels(n_samples, data: pd.DataFrame, labels: pd.DataFrame):
    """
    Returns sample from the data based on the data labes.
    :param n_samples: Specify the number of samples for each label. If greater than count of objects for the particular
    label, will be set to the count of the objects.
    :param data: Data to generate samples from
    :param labels: Data labels
    :return: Samples taken from the data and their labels.
    """
    logger.info("Getting samples from the dataset, number of samples for each label is set to {0}.", n_samples)

    labels_counts = labels.value_counts().apply(lambda x: x if x <= n_samples else n_samples)
    total_samples = labels_counts.sum()

    logger.info("Found {0} target classes, {1} samples at total.", labels_counts.index.shape[0], total_samples)

    samples_idx = np.array([])
    for target_class in labels_counts.index:
        indexes_class = labels[labels == target_class].index.to_numpy()
        class_idx = np.random.RandomState(seed=42).permutation(indexes_class)[0:labels_counts[target_class]]
        samples_idx = np.concatenate((samples_idx, class_idx), axis=0)

    samples = data.loc[samples_idx.tolist()]
    samples_labels = labels.loc[samples_idx.tolist()]
    return samples, samples_labels
