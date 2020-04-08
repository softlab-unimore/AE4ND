import os
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

from scipy import stats

from utils.tools import get_settings


def get_arguments():
    parser = argparse.ArgumentParser(description='Run "One Threshold Algorithm"')
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help='file to analise')

    parser.add_argument('--settings',
                        type=str,
                        required=False,
                        help='settings to provide the setup starting time')

    parser.add_argument('--sep',
                        type=str,
                        default=';',
                        help='table separator')

    parser.add_argument('--save',
                        action='store_true',
                        help='to save the final algorithm result')

    args = parser.parse_args()
    if not os.path.isfile(args.file):
        raise ValueError('{} is not a valid file'.format(args.file))

    if args.settings and not os.path.isfile(args.settings):
        raise ValueError('{} is not a valid settings file'.format(args.settings))

    return args


def one_threshold(s, start, th, size, margin):
    s = s[s.index >= start]

    start = s.index.min()
    if s.empty:
        print('s is empty')
        return None

    if np.std(s.values) == 0:
        print('s std is zero')
        return None

    # z-normalization on s
    s[:] = stats.zscore(s.values)

    # keep only not nan value
    cond = np.isnan(s).any()

    if cond:
        print('nan value in series')
        return start, start

    size_support = size
    margin_support = margin
    time = None
    reset = True

    for idx, val in s.iteritems():
        if reset:
            size_support = size
            margin_support = margin
            time = idx
            reset = False

        if abs(val) <= th:
            size_support -= 1
        elif margin_support > 0:
            size_support -= 1
            margin_support -= 1
        else:
            reset = True

        if size_support == 0:
            break

    return start, time


def main():
    args = get_arguments()
    ds = pd.read_csv(args.file, args.sep)
    if 'DT' not in ds.columns:
        raise ValueError('file doesn\'t contain DT column for datetime index')

    ds.set_index('DT', inplace=True)
    ds.sort_index(inplace=True)
    ds.index = pd.to_datetime(ds.index)

    start_time = ds.index.min()
    if args.settings:
        settings = pd.read_csv(args.settings, sep=args.sep)
        settings.set_index('DT', inplace=True)
        settings.index = pd.to_datetime(settings.index)
        if 'ltime' in settings.columns:
            settings.ltime = pd.to_datetime(settings.ltime)

        if 'rtime' in settings.columns:
            settings.rtime = pd.to_datetime(settings.rtime)


        setting = get_settings(ds, settings)
        if 'ltime' in setting:
            start_time = setting.ltime
        else:
            start_time = setting.name

    results = []
    for col in ds.columns:
        time = one_threshold(ds[col].copy(),
                             start=start_time,
                             th=0.5, size=800, margin=1)
        if time and time[0] != time[1]:
            results.append({'feature': col, 'start': time[0], 'end': time[1]})

    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    if args.save:
        output_dir = 'results'
        filename = os.path.basename(args.file)
        filename = os.path.join(output_dir, 'one_threshold_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, args.sep, index=False)


if __name__ == '__main__':
    main()
