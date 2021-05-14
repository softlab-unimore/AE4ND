import os
import json
import argparse
import numpy as np
import pandas as pd

from timely.utils.manage_model import get_model
from timely.utils.manage_file import get_files, read_ds_lvm
from timely.utils.tools import get_sliding_window_matrix

from timely.transforms.transformer import resample


def get_argument():
    parser = argparse.ArgumentParser(description='FULL Evaluation Anomaly and Normal behaviour')

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

    size = 600

    features_list = [
        "Acceleration_X1",
        "Acceleration_Y1",
        "Acceleration_Z1",
        "Acceleration_X2",
        "Acceleration_Y2",
        "Acceleration_Z2",
        "Acceleration_X3",
        "Acceleration_Y3",
        "Acceleration_Z3"
    ]

    stride = 1

    model_list = [
        'cnn',
        'lstm',
        'deep',
        'isolation_forest',
        'setup_clustering',
        'pca',
        'lof',
        'svm',
    ]

    kernel_list = [180 if model_type in ['cnn', 'lstm', 'deep'] else 10 for model_type in model_list]

    resample_rate = 6400

    save_result = True
    output_dir = './results'

    # Initialize result array to memorize result
    # for each train and test step
    result_array = []

    # Get files from selected folder to use for training and testing
    curr_files = []
    for folder in all_state_folder:
        curr_files += get_files(folder, ext='lvm')[:]

    test_files = curr_files

    for model_type, kernel in zip(model_list, kernel_list):
        print('\n' + '\\\\//' * 20)
        print('\n Model: {}\n'.format(model_type))

        params_file = './params/params_{}.json'.format(model_type)

        for train_file in curr_files:

            train_state = os.path.split(os.path.dirname(train_file))[-1]
            print("\n State Train: ", train_state)
            print("Read Train File: ", os.path.basename(train_file))
            ds_train = read_ds_lvm(train_file, get_header=False)

            # Check train
            if ds_train is None or ds_train.empty:
                print('Impossible read train file')
                continue

            # Select features
            ds_train = ds_train[features_list]

            # Resample
            train_len = len(ds_train)
            ds_train = resample(ds_train, resample_rate)
            # ds_train = ds_train[:num_sample]
            print('Train Original File Length: ', train_len)
            print('New File Length {} {:.02f}'.format(len(ds_train), 100 * len(ds_train) / train_len))

            # Create training set
            print("Create training set")
            x_train = get_sliding_window_matrix(ds_train.values, kernel, stride)
            x_train = x_train[:size]
            train_len = len(x_train)
            print('Train shape ', x_train.shape)


            # Model initialization
            print("Model initialization: {}".format(model_type))
            model = get_model(model_type, params_file=params_file)

            # Training
            print("Training...")
            model.fit(x_train)

            for test_file in test_files:

                test_state = os.path.split(os.path.dirname(test_file))[-1]

                if train_state == test_state \
                        and test_file == train_file:
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
                ds_test = resample(ds_test, resample_rate)
                # ds_test = ds_test[:num_sample]
                print('Test Original File Length: ', test_len)
                print('New File Length {} {:.02f}'.format(len(ds_test), 100 * len(ds_test) / test_len))

                test_stride = 1

                # Create set
                print("Create testing set")
                x_test = get_sliding_window_matrix(ds_test.values, kernel, test_stride)
                print('Test shape ', x_test.shape)

                # Testing
                print('Testing...')
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
                    'TRAIN_STATE': train_state,
                    'TRAIN': os.path.basename(train_file),
                    'TRAIN_SIZE': train_len,
                    'TEST_STATE': test_state,
                    'TEST': os.path.basename(test_file),
                    'TEST_LEN': test_len,
                    'NUM_SINGLE_ANOMALY': num_error,
                    'PCT_ANOMALY': mean_error,
                    'NUM_SAMPLE_ANOMALY': mean_only_error,
                    'NUM_SAMPLE': len(x_test),
                    'LABEL': train_state != test_state
                }

                result_array.append(result_record)

        if save_result:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            filename = os.path.join(output_dir, 'results_single_' + model_type + '.csv')

            result_ds = pd.DataFrame(result_array)

            result_ds.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
