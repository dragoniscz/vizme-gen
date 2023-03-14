import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.select_dtypes(include='number')
    subset = (subset - subset.min()) / (subset.max() - subset.min())
    df[subset.columns] = subset
    return df


def quantize(df: pd.DataFrame, n_quantiles=1000, output_distribution="uniform", random_state=0) -> pd.DataFrame:
    subset = df.select_dtypes(include='number')
    from sklearn.preprocessing import quantile_transform
    transformed = quantile_transform(subset, n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=random_state)
    df[subset.columns] = transformed
    return df
