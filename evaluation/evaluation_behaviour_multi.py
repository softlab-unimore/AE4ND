import os
import json
import argparse
import numpy as np
import pandas as pd

from datetime import datetime

from utils.manage_model import get_model, predict_anomaly
from utils.manage_file import get_files, read_ds_lvm
from utils.tools import create_triplet_time_series, get_sliding_window_matrix

from transforms.transformer import resample, resample_with_feature_extractor, get_transformer


def get_argument():
    parser = argparse.ArgumentParser(description='Evaluation Anomaly and Normal behaviour')

    parser.add_argument('--input', required=True, type=str, help='Input Params')
    args = parser.parse_args()

    try:
        with open(args.input) as file:
            params = json.load(file)
    except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError('Impossible read json params')

    return params


def main():
    params = get_argument()
    all_state_folder = params['all_state_folder']
    features_list = params['features_list']
    kernel = params['kernel']
    stride = params['stride']
    model_type = params['model_type']
    resample_rate = params.get('resample_rate', 6400)
    with_decision_score = params.get('with_decision_score', False)
    custom_resample = params.get('custom_resample', False)

    # resample_rate = 12800  # 12800 sample are 1 second
    # num_sample = 1000000
    with_skip = False

    params_file = './params/params_{}.json'.format(model_type)
    save_result = True
    overwrite = True
    output_dir = './results'

    result_array = []

    # Get list of list of files, where for each state we have a list of file
    curr_files = []

    # Get list of test files
    test_files = []

    for folder in all_state_folder:
        files = get_files(folder, ext='lvm')
        curr_files.append(files)
        test_files += files

    max_size = min([len(files) for files in curr_files[:3]])

    # Get train files where each element is a list of files for a single train
    train_files = []
    for i in range(max_size):
        train_pack = [files[i] for files in curr_files[:3]]

        for j in range(1, len(train_pack)):
            train_files.append(train_pack[:j + 1])

    for train_pack in train_files:
        print('\n' + '\\\\//' * 20)

        selected_files = []
        train_states = []
        x_states = []

        print('\n Train Pack')
        for train_file in train_pack:
            train_state = os.path.split(os.path.dirname(train_file))[-1]
            print("State: ", train_state)
            print("Read File: ", os.path.basename(train_file))
            ds_train = read_ds_lvm(train_file, get_header=False)

            # Check train
            if ds_train is None or ds_train.empty:
                print('Impossible read train file')
                continue

            # Select features
            ds_train = ds_train[features_list]

            # Resample
            train_len = len(ds_train)
            if custom_resample:
                ds_train = resample_with_feature_extractor(ds_train, resample_rate)
            else:
                ds_train = resample(ds_train, resample_rate)

            # ds_train = ds_train[:num_sample]
            print('Original File Length: ', train_len)
            print('New File Length {} {:.02f}'.format(len(ds_train), 100 * len(ds_train) / train_len))

            # Create training set
            print("Create set")
            x_train = get_sliding_window_matrix(ds_train.values, kernel, stride)
            print('Shape ', x_train.shape)

            selected_files.append(train_file)
            train_states.append(train_state)
            x_states.append(x_train)

        x_states = np.vstack(x_states)
        print('\n Train Size: ', x_states.shape)
        print('Train state: ', train_states)

        # Model initialization
        print("Model initialization: {}".format(model_type))
        model = get_model(model_type, params_file=params_file)

        # Training
        print("Training...")
        model.fit(x_states)

        for test_file in test_files:

            test_state = os.path.split(os.path.dirname(test_file))[-1]

            if test_file in selected_files:
                continue

            print("\n State Test: ", test_state)
            print("Read Test File: ", os.path.basename(test_file))
            ds_test = read_ds_lvm(test_file, get_header=False)

            # t1 = datetime.now()

            # Check test
            if ds_test is None or ds_test.empty:
                print('Impossible read test file')
                continue

            # Select features
            ds_test = ds_test[features_list]

            # Resample
            test_len = len(ds_test)
            if custom_resample:
                ds_test = resample_with_feature_extractor(ds_test, resample_rate)
            else:
                ds_test = resample(ds_test, resample_rate)
            # ds_test = ds_test[:num_sample]
            print('Test Original File Length: ', test_len)
            print('New File Length {} {:.02f}'.format(len(ds_test), 100 * len(ds_test) / test_len))

            if with_skip:
                test_stride = kernel
            else:
                test_stride = 1

            # Create set
            print("Create testing set")
            x_test = get_sliding_window_matrix(ds_test.values, kernel, test_stride)
            print('Test shape ', x_test.shape)

            # Testing
            print('Testing...')
            if with_decision_score:
                y_pred = model.decision_score(x_test)
            else:
                y_pred = model.predict(x_test)

            num_error = np.sum(y_pred > 0)
            mean_error = np.mean(y_pred)
            if num_error > 0:
                mean_only_error = np.mean(y_pred[y_pred > 0])
            else:
                mean_only_error = 0

            if not np.sum(y_pred > 0):
                print("Results: NO Anomaly founded")
            else:
                print("Results: {} anomalies "
                      "({:.05f} total {})".format(
                    num_error, mean_error, len(x_test)))

            result_record = {
                'MODEL': model_type,
                'KERNEL': kernel,
                'STRIDE': stride,
                'TRAIN_STATE': train_states,
                'TRAIN': [os.path.basename(train_file) for train_file in selected_files],
                'TEST_STATE': test_state,
                'TEST': os.path.basename(test_file),
                'NUM_SINGLE_ANOMALY': num_error,
                'PCT_ANOMALY': mean_error,
                'NUM_SAMPLE_ANOMALY': mean_only_error,
                'NUM_SAMPLE': len(x_test),
                'LABEL': test_state not in train_states
            }

            result_array.append(result_record)

    if save_result:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, 'results_multi_' + model_type + '.csv')

        result_ds = pd.DataFrame(result_array)

        if os.path.isfile(filename) and not overwrite:
            prev_result_ds = pd.read_csv(filename)
            result_ds = pd.concat([prev_result_ds, result_ds], axis=0, ignore_index=True)

        result_ds.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
