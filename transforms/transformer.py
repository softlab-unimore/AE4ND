import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def moving_average(ds: pd.DataFrame, window: int = 7):
    return ds.rolling(window=window).mean()


def exponential_moving_average(ds: pd.DataFrame, window: int = 7):
    return ds.ewm(span=window, adjust=True).mean()


def resample(ds: pd.DataFrame, rate: int):
    new_index = np.arange(len(ds))
    new_index = new_index // rate
    return ds.groupby(new_index).mean()


def resample_custom(ds: pd.DataFrame, rate: int, custom):
    new_index = np.arange(len(ds))
    new_index = new_index // rate

    return ds.groupby(new_index).agg(custom)


def get_transformer(ds: pd.DataFrame, transform_type: str):
    if transform_type == 'std':
        transformer = StandardScaler()
    elif transform_type == 'minmax':
        transformer = MinMaxScaler()
    elif transform_type == 'robust':
        transformer = RobustScaler()
    else:
        raise ValueError('Select the wrong transformer: ', transform_type)

    transformer.fit(ds)
    return transformer
