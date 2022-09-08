import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .extractor import *


def moving_average(df: pd.DataFrame, window: int = 7):
    return df.rolling(window=window).mean()


def exponential_moving_average(df: pd.DataFrame, window: int = 7):
    return df.ewm(span=window, adjust=True).mean()


def resample(df: pd.DataFrame, rate: int):
    new_index = np.arange(len(df))
    new_index = new_index // rate
    return df.groupby(new_index).mean()


def resample_custom(df: pd.DataFrame, rate: int, custom):
    new_index = np.arange(len(df))
    new_index = new_index // rate

    return df.groupby(new_index).agg(custom)


def resample_with_feature_extractor(df: pd.DataFrame, rate: int):
    custom_function = {
        'mean': mean,
        'peak': peak,
        'peak_to_peak': peak_to_peak,
        'rms': rms,
        'crest_factor': crest_factor,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shape_factor': shape_factor
    }

    res_df = []
    for key, custom in custom_function.items():
        new_df = resample_custom(df, rate, custom)
        new_df.columns = ['{}_{}'.format(key, col) for col in new_df.columns]

        res_df.append(new_df)

    res_df = pd.concat(res_df, axis=1)
    return res_df


def get_transformer(transform_type: str):
    if transform_type == 'std':
        transformer = StandardScaler()
    elif transform_type == 'minmax':
        transformer = MinMaxScaler()
    elif transform_type == 'robust':
        transformer = RobustScaler()
    else:
        raise ValueError('Select the wrong transformer: ', transform_type)
    return transformer


def apply_transform(df, transformer):
    """ Transform a dataframe by applying a trained transformer """
    return pd.DataFrame(transformer.transform(df.values), columns=df.columns)


def fit_and_transform_data_list(df_list, transform_type):
    """ Train a transformer on a given list of time series """
    # Concatenate a list of dataframes along temporal axis
    df_concat = pd.concat(df_list, axis=0, ignore_index=True)

    # Get a trained transformer on the concatenated dataframes
    transformer = get_transformer(transform_type)
    transformer.fit(df_concat.values)

    # Apply transformer for each raw dataframes
    df_transformed_list = [apply_transform(df, transformer) for df in df_list]

    return df_transformed_list, transformer


def transform_data_list(df_list, transformer):
    """ Apply a trained transformer on a list of time series """
    # Apply transformer for each raw dataframes
    df_transformed_list = [apply_transform(df, transformer) for df in df_list]

    return df_transformed_list
