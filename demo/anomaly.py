import os
import json
import argparse
import pandas as pd
from tabulate import tabulate

from timely.utils.manage_model import get_model, predict_anomaly
from timely.utils.tools import create_triplet_time_series, get_sliding_window_matrix

from timely.transforms.transformer import resample, resample_with_feature_extractor

info = """
Please  provide the following information in json format: 

train:              File used to train the model 
test:               File used to test and find anomalies 
features_list:      The features selected in the provided files ([] all features are used) 
kernel:             Windows size 
stride:             Value of sliding windows 
resample_rate:      Resample frequency to apply an average windows if custom_resample is False, 
custom_resample:    True apply a custom resample strategy by extracting features from ae4nd domain 
model_type:         Model selected for anomaly detection: 
                        DL based: cnn, deep, lstm 
                        ML based: isolation_forest, setup_clustering, pca, svm, lof 
model_params:       Params used to train the model if empty the default configuration is used 
"""


def get_argument():
    parser = argparse.ArgumentParser(description='Anomaly Detection Algorithms', epilog=info)

    parser.add_argument('--params', required=True, type=str, help='input json file')
    args = parser.parse_args()

    try:
        with open(args.params) as file:
            params = json.load(file)
    except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError('Impossible read json params')

    arr = ["train",
           "test",
           "features_list",
           "kernel",
           "stride",
           "resample_rate",
           "custom_resample",
           "model_type",
           "model_params"]

    for val in arr:
        if val not in params:
            raise ValueError('No {} selected please provide:'.format(val))

    if not os.path.isfile(params['train']) or not os.path.isfile(params['test']):
        raise ValueError('Please select valid file')

    return params


def main():
    params = get_argument()

    # model input
    train_file = params['train']
    test_file = params['test']

    # feature params
    features_list = params['features_list']
    kernel = params['kernel']
    stride = params['stride']

    # feature extraction
    resample_rate = params.get('resample_rate', 6400)
    custom_resample = params.get('custom_resample', False)

    # model params
    model_type = params['model_type']
    params_file = params['model_params']

    # Read train file
    print("Read Train File: ", os.path.basename(train_file))
    ds_train = pd.read_csv(train_file)

    # Select features
    if features_list:
        ds_train = ds_train[features_list]

    # Resample
    train_len = len(ds_train)
    if custom_resample:
        ds_train = resample_with_feature_extractor(ds_train, resample_rate)
    else:
        if resample_rate > 1:
            ds_train = resample(ds_train, resample_rate)

    print('Train Original File Length: ', train_len)
    print('New File Length {} {:.02f}'.format(len(ds_train), 100 * len(ds_train) / train_len))

    # Create training set
    print("Create training set")
    x_train = get_sliding_window_matrix(ds_train.values, kernel, stride)
    print('Train shape ', x_train.shape)

    # Model initialization
    print("Model initialization: {}".format(model_type))
    model = get_model(model_type, params_file=params_file)

    # Training
    print("Training...")
    model.fit(x_train)

    print("Read Test File: ", os.path.basename(test_file))
    ds_test = pd.read_csv(test_file)

    # Select features
    if features_list:
        ds_test = ds_test[features_list]

    # Resample
    test_len = len(ds_test)
    if custom_resample:
        ds_test = resample_with_feature_extractor(ds_test, resample_rate)
    else:
        if resample_rate > 1:
            ds_test = resample(ds_test, resample_rate)

    print('Test Original File Length: ', test_len)
    print('New File Length {} {:.02f}'.format(len(ds_test), 100 * len(ds_test) / test_len))

    print('Testing...')
    y_pred = predict_anomaly(ds_test, model, kernel, with_skip=False)

    # Encoding results into triplet formats
    results = create_triplet_time_series(y_pred, with_support=True)

    # Show results
    print("Results:")
    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main()
