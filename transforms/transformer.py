import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def moving_average(ds: pd.DataFrame, window: int = 7):
    return ds.rolling(window=window).mean()


def exponential_moving_average(ds: pd.DataFrame, window: int = 7):
    return ds.ewm(span=window, adjust=True).mean()


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
