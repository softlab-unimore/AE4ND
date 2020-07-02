import os
import json
import argparse
import pandas as pd
from tabulate import tabulate

from utils.tools import get_settings

from model.OneThreshold.one_threshold import OneThreshold


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


def main():
    args = get_arguments()

    # read dataset
    ds = pd.read_csv(args.file, args.sep)
    if 'DT' not in ds.columns:
        raise ValueError('file doesn\'t contain DT column for datetime index')

    # set time index
    ds.set_index('DT', inplace=True)
    ds.index = pd.to_datetime(ds.index)
    ds.sort_index(inplace=True)

    # get model params
    with open("../params/params_one_threshold.json") as file:
        params = json.load(file)

    # define model
    model = OneThreshold(th=params['th'], size=params['size'], margin=params['margin'])

    # define the setup start
    start_time = ds.index.min()
    if args.settings:
        # get the setup start from settings
        settings = pd.read_csv(args.settings, sep=args.sep)
        settings.set_index('DT', inplace=True)
        settings.index = pd.to_datetime(settings.index)

        # convert ltime and rtime column into datetime values
        if 'ltime' in settings.columns:
            settings.ltime = pd.to_datetime(settings.ltime)
        if 'rtime' in settings.columns:
            settings.rtime = pd.to_datetime(settings.rtime)

        # get setting for the given dataset
        setting = get_settings(ds, settings)
        if 'ltime' in setting:
            start_time = setting.ltime
        else:
            start_time = setting.name

    results = []
    for col in ds.columns:
        time = model.predict(ds[col].copy(), start=start_time)

        if time and time[0] != time[1]:
            results.append({'feature': col, 'start': time[0], 'end': time[1]})

    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    if args.save:
        output_dir = '../results'
        filename = os.path.basename(args.file)
        filename = os.path.join(output_dir, 'one_threshold_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, args.sep, index=False)


if __name__ == '__main__':
    main()
