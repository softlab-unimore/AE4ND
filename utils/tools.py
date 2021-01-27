import numpy as np
import pandas as pd


def label_settings(settings, columns):
    """identify and label unique settings"""
    lsettings = settings[columns].drop_duplicates()
    lsettings.index = range(len(lsettings))

    settings_map = {}
    for idx, val in lsettings.iterrows():
        settings_map[tuple(val.values.tolist())] = idx

    settings['label'] = settings.apply(lambda x: settings_map[tuple(x[columns].tolist())], axis=1)


def get_settings(ds, settings):
    """get the settings for a given chunk file"""
    min_idx = settings.index[settings.index <= ds.index.min()].max()
    # max_idx = settings.index[settings.index >= ds.index.max()].min()
    s = settings.loc[min_idx]

    if 'ltime' in s and s.ltime > ds.index.max():
        # print('Change settings and jump into next setting {} > {}'.format(s.ltime, ds.index.max()))
        # min_idx = settings.index[settings.index < min_idx].max()
        # s = settings.loc[min_idx]
        pass

    return s


def check_setup(ds, setting):
    t1 = ds.index.min()
    t2 = ds.index.max()

    s1 = setting.ltime
    s2 = setting.rtime

    if t1 < s2 and t2 > s1:
        return True

    return False


def lazy_check_setup(ds, setting):
    t1 = ds.index.min()

    s1 = setting.name

    if (t1 - s1).days == 0:
        return True

    return False


def create_triplet_time_series(ts: pd.Series, with_support: bool = False):
    """
    create triplet time series encoding
    withSupport if return the number of compressed records
    """
    res = []
    start = -1
    prev = -1
    support = 0
    for k, val in ts.iteritems():
        support += 1
        if start == -1 and val > 0:
            start = k
            support = 0
        elif start >= 0 and val == 0:
            x = {
                'feature': ts.name,
                'start': start,
                'end': prev
            }
            if with_support:
                x['support'] = support
            res.append(x)
            start = -1

        prev = k

    if start != -1:
        x = {
            'feature': ts.name,
            'start': start,
            'end': prev
        }
        if with_support:
            x['support'] = support
        res.append(x)

    return res


def get_time_series_dataset(filename: str, sep: str, col: str):
    """get time series dataframe dataset"""
    # Get DataFrame from filename
    try:
        ds = pd.read_csv(filename, sep=sep, encoding='latin1')
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
