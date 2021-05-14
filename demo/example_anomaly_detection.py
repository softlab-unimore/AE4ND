import os
import json
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
# import matplotlib.pyplot as plt

from timely.models.anomaly_detection.setup_clustering import SetupClustering
from timely.models.anomaly_detection.isolation_forest import IsolationForest
from timely.models.anomaly_detection.one_class_svm import OneClassSVM
from timely.models.anomaly_detection.local_outlier_factor import LOF
from timely.models.anomaly_detection.pca import PCA

from timely.utils.tools import create_triplet_time_series, get_time_series_dataset, get_sliding_window_matrix
from timely.utils.manage_model import get_model

# Input files
train_file = './data/industrial/ts_normal1.CSV'
test_file = './data/industrial/ts_anomaly_setup1.CSV'
datetime_col = 'DT'
sep = ';'

# Empty list will select all input columns like features
features_list = [
    '%E1', 'TEMP_E1',
    '%E2', 'TEMP_E2',
    '%E3', 'TEMP_E3',
    '%E4', 'TEMP_E4',
    '%E5', 'TEMP_E5',
]

# Data preparation params
kernel = 60
stride = 10

# Isolation Forest params
model_type = 'pca'
params_file = None

# Demo params
save = False
save_model = False
visualize = False
output_dir = './results'


def get_model(model_type, params_file=None):
    # Get models params
    # print('Get params')
    if params_file is None:
        params = {}
    else:
        try:
            with open(params_file) as file:
                params = json.load(file)
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError('Impossible read json params')

    # Model initialization
    # print("Model initialization")
    if model_type == 'isolation_forest':
        model = IsolationForest(verbose=True, n_jobs=-1, **params)
    elif model_type == 'setup_clustering':
        model = SetupClustering(**params)
    elif model_type == 'pca':
        model = PCA(**params)
    elif model_type == 'svm':
        model = OneClassSVM(**params)
    elif model_type == 'lof':
        model = LOF(**params)
    else:
        raise ValueError('{} does not exist'.format(model_type))

    return model


def predict_anomaly(ds, model, kernel, with_skip=True):
    if with_skip:
        stride = kernel
    else:
        stride = 1

    # Create set
    print("Create testing set")
    x_test = get_sliding_window_matrix(ds.values, kernel, stride)

    # Testing
    print('Testing...')
    y_pred = model.predict(x_test)

    # Expand results
    y_pred = [val for val in y_pred for _ in range(stride)]
    res = np.zeros((len(ds)))

    if with_skip:
        res[:len(y_pred)] = y_pred
    else:
        res[-len(y_pred):] = y_pred

    y_pred = pd.Series(res, index=ds.index, name='features')

    return y_pred


def main():
    print('Read input data')

    # Get train dataset
    print('train: {}'.format(train_file))
    ds_train = get_time_series_dataset(filename=train_file, sep=sep, col=datetime_col)
    # Check train
    if ds_train is None:
        raise ValueError('Impossible read train file')

    # Get test dataset
    print('test: {}'.format(test_file))
    ds_test = get_time_series_dataset(filename=test_file, sep=sep, col=datetime_col)
    # Check test
    if ds_test is None:
        raise ValueError('Impossible read test file')
    print('from {} to {}'.format(ds_test.index.min(), ds_test.index.max()))

    # Get features
    print('Select features')
    features = features_list
    if not features:
        features = ds_train.columns.to_list()
    elif set(features).difference(set(ds_train.columns)):
        raise ValueError('Select the wrong features')

    assert np.all(ds_train.columns == ds_test.columns), 'Train and Test file have different features'

    # Select features
    ds_train = ds_train[features]
    ds_test = ds_test[features]

    # if visualize:
    #     fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    #     ds_train.plot(ax=ax[0])
    #     ax[0].set_title('Training Data')
    #
    #     ds_test.plot(ax=ax[1])
    #     ax[1].set_title('Test Data')
    #     plt.show()

    # Model initialization
    print("Model initialization: {}".format(model_type))
    model = get_model(model_type, params_file=params_file)

    # Create training set
    print("Create training set")
    x_train = get_sliding_window_matrix(ds_train.values, kernel, stride)

    # Training
    print("Training...")
    model.fit(x_train)

    # Option 1: Save trained models
    if save_model:
        # Create output directory
        filename = os.path.join(output_dir, 'model_{}.pkl'.format(model_type))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save trained models
        joblib.dump(model, filename)

        # Load trained models
        model = joblib.load(filename)

    # Testing
    print('Testing...')
    y_pred = predict_anomaly(ds_test, model, kernel, with_skip=False)

    # Encoding results into triplet formats
    results = create_triplet_time_series(y_pred, with_support=True)

    # Show results
    print("Results:")
    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    # Save results
    if save:
        filename = os.path.basename(test_file)
        filename = os.path.join(output_dir, 'results_' + model_type + '_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, sep=sep, index=False)


if __name__ == '__main__':
    main()
