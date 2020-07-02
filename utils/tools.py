import numpy as np
import pandas as pd


def get_settings(ds, settings):
    """get the settings for a given chunk file"""
    min_idx = settings.index[settings.index <= ds.index.min()].max()
    # max_idx = settings.index[settings.index >= ds.index.max()].min()
    s = settings.loc[min_idx]

    if 'ltime' in s and s.ltime > ds.index.max():
        min_idx = settings.index[settings.index < min_idx].max()
        s = settings.loc[min_idx]

    return s


def create_triplet_time_series(ts: pd.Series):
    """create triplet time series encoding"""
    res = []
    start = None
    prev = None
    for k, val in ts.iteritems():
        if start is None and val > 0:
            start = k
        elif start and val == 0:
            res.append({
                'feature': ts.name,
                'start': start,
                'end': prev
            })
            start = None

        prev = k
    return res


def get_time_series_dataset(filename: str, sep: str, col: str):
    """get time series dataframe dataset"""
    # Get DataFrame from filename
    try:
        ds = pd.read_csv(filename, sep=sep)
    except (pd.errors.ParserError, FileNotFoundError):
        return None

    # Check datetime column col
    if col not in ds.columns:
        return None

    # Create datetime index
    ds.set_index('DT', inplace=True)
    ds.sort_index(inplace=True)
    ds.index = pd.to_datetime(ds.index)

    return ds


def get_sliding_window_matrix(data, window, stride):
    """convert input matrix using sliding window"""
    rows, cols = data.shape
    new_rows = 1 + (rows - window) // stride

    matrix = np.zeros((new_rows, window, cols))

    for i in range(new_rows):
        left = i * stride
        right = left + window
        matrix[i, :, :] = data[left:right, :]

    return matrix
