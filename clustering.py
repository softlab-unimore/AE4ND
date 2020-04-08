import os
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

from model.setup_clustering import SetupClustering


def get_arguments():
    parser = argparse.ArgumentParser(description='Run "Setup Clustering Algorithm"')
    parser.add_argument('--test',
                        type=str,
                        required=True,
                        help='file to analyze and detect abnormal condition')

    parser.add_argument('--train',
                        type=str,
                        required=True,
                        help='file used to learn the normal state')

    parser.add_argument('--sep',
                        type=str,
                        default=';',
                        help='table separator')

    parser.add_argument('--single',
                        action='store_true',
                        help='learn normal state for each feature, or learn a general normal state for all features')

    parser.add_argument('--save',
                        action='store_true',
                        help='to save the final algorithm result')

    args = parser.parse_args()
    if not os.path.isfile(args.test):
        raise ValueError('{} is not a valid test'.format(args.test))

    if not os.path.isfile(args.train):
        raise ValueError('{} is not a valid train'.format(args.train))

    return args


def compact_time_series(ts: pd.Series):
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


def main():
    args = get_arguments()
    ds_train = pd.read_csv(args.train, args.sep)
    if 'DT' not in ds_train.columns:
        raise ValueError('train file doesn\'t contain DT column for datetime index')

    ds_train.set_index('DT', inplace=True)
    ds_train.sort_index(inplace=True)
    ds_train.index = pd.to_datetime(ds_train.index)

    ds_test = pd.read_csv(args.test, args.sep)
    if 'DT' not in ds_test.columns:
        raise ValueError('test file doesn\'t contain DT column for datetime index')

    ds_test.set_index('DT', inplace=True)
    ds_test.sort_index(inplace=True)
    ds_test.index = pd.to_datetime(ds_test.index)

    window_width = 200
    height = len(ds_train) // window_width

    # Train phase
    model = SetupClustering(max_dist=0.001, anomaly_threshold=0.00001)
    models = {}
    if args.single:
        for col in ds_train.columns:
            models[col] = SetupClustering(max_dist=0.001, anomaly_threshold=0.0001)
            x_train = ds_train[col].iloc[: height * window_width].values
            models[col].fit(np.reshape(x_train, (height, window_width)))
    else:
        x_train = []
        for col in ds_train.columns:
            block = ds_train[col].iloc[: height * window_width].values
            block = np.reshape(block, (height, window_width))
            x_train.append(block)

        x_train = np.concatenate(x_train, axis=0)
        model.fit(x_train)

    results = []
    height = len(ds_test) // window_width
    for col in ds_test.columns:
        x_test = ds_test[col].iloc[: height * window_width].values
        x_test = np.reshape(x_test, (height, window_width))

        if len(ds_test) % window_width != 0:
            margin = ds_test[col].iloc[-window_width:].values
            x_test = np.concatenate([x_test, [margin]], axis=0)

        if args.single:
            y_pred = models[col].predict(x_test)
        else:
            y_pred = model.predict(x_test)

        # expand results
        y_pred = [val for val in y_pred for _ in range(window_width)]
        y_pred = y_pred[:len(ds_test)]
        y_pred = pd.Series(y_pred, index=ds_test.index, name=col)

        results += compact_time_series(y_pred)

    # Show results
    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    if args.save:
        output_dir = 'results'
        filename = os.path.basename(args.file)
        filename = os.path.join(output_dir, 'clustering_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, args.sep, index=False)


if __name__ == '__main__':
    main()
