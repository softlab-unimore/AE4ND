import os
import json
import joblib
import argparse

import numpy as np
import pandas as pd
from tabulate import tabulate
from model.SetupClustering.setup_clustering import SetupClustering
from utils.tools import create_triplet_time_series, get_time_series_dataset, get_sliding_window_matrix


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
                        help='table separator to analyze input dataset')

    parser.add_argument('--features',
                        nargs='+',
                        help='feature list where the algorithm is applied')

    parser.add_argument('--save',
                        action='store_true',
                        help='to save the final algorithm result')

    args = parser.parse_args()
    if not os.path.isfile(args.test):
        raise ValueError('{} is not a valid test'.format(args.test))

    if not os.path.isfile(args.train):
        raise ValueError('{} is not a valid train'.format(args.train))

    return args


def main():
    args = get_arguments()

    print('Read input data')
    # Get train dataset
    ds_train = get_time_series_dataset(filename=args.train, sep=args.sep, col='DT')
    # Check train
    if ds_train is None:
        raise ValueError('Impossible read train file')

    # Get test dataset
    ds_test = get_time_series_dataset(filename=args.test, sep=args.sep, col='DT')
    # Check test
    if ds_test is None:
        raise ValueError('Impossible read test file')

    # Get model params
    print('Get params')
    try:
        with open("./params/params_clustering.json") as file:
            params = json.load(file)
    except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError('Impossible read json params')

    # Get features
    print('Select features')
    features = args.features
    if not features:
        features = ds_train.columns.to_list()
    elif set(features).difference(set(ds_train.columns)):
        raise ValueError('Select the wrong features')

    # Select features
    ds_train = ds_train[features]
    ds_test = ds_test[features]

    # Model initialization
    print("Model initialization")
    model = SetupClustering(
        distance=params['distance'],
        max_dist=params['max_distance'],
        anomaly_threshold=params['anomaly_threshold'])

    # Create training set
    print("Create training set")
    x_train = get_sliding_window_matrix(ds_train.values,
                                        params['kernel'],
                                        params['stride'])

    # Training
    print("Training...")
    model.fit(x_train)

    # Option 1: save trained model
    # Save trained model
    # joblib.dump(model, 'results/model.pkl')
    # Load trained model
    # model = joblib.load('results/model.pkl')

    # Option 2: save model state
    # res = model.save_model(filename='results/data.pkl')
    # if not res:
    #     raise ValueError('Impossible save model internal state')
    # model.load_model(filename='results/data.pkl')

    # Create test set
    print("Create testing set")
    x_test = get_sliding_window_matrix(ds_test.values,
                                       params['kernel'],
                                       params['kernel'])

    # Testing
    print('Testing...')
    y_pred = model.predict(x_test)

    # Expand results
    y_pred = [val for val in y_pred for _ in range(params['kernel'])]
    res = np.zeros((len(ds_test)))
    res[:len(y_pred)] = y_pred
    y_pred = pd.Series(res, index=ds_test.index, name='features')

    # Encoding results into triplet formats
    results = create_triplet_time_series(y_pred)

    # Show results
    print("Results:")
    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    # Save results
    if args.save:
        output_dir = '../results'
        filename = os.path.basename(args.test)
        filename = os.path.join(output_dir, 'clustering_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, sep=args.sep, index=False)


if __name__ == '__main__':
    main()
