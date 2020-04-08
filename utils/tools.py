import os
import pandas as pd


# get the settings for a given chunk file
def get_settings(ds, settings):
    min_idx = settings.index[settings.index <= ds.index.min()].max()
    # max_idx = settings.index[settings.index >= ds.index.max()].min()
    s = settings.loc[min_idx]

    if 'ltime' in s and s.ltime > ds.index.max():
        min_idx = settings.index[settings.index < min_idx].max()
        s = settings.loc[min_idx]

    return s
