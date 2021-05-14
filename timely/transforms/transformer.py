import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .extractor import *


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


def resample_with_feature_extractor(ds: pd.DataFrame, rate: int):
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

    res_ds = []
    for key, custom in custom_function.items():
        new_ds = resample_custom(ds, rate, custom)
        new_ds.columns = ['{}_{}'.format(key, col) for col in new_ds.columns]

        res_ds.append(new_ds)

    res_ds = pd.concat(res_ds, axis=1)
    return res_ds


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
