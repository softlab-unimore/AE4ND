import os
import json
import argparse
import pandas as pd

from utils.manage_model import get_model, predict_anomaly
from utils.manage_file import get_files, read_ds_lvm
from utils.tools import create_triplet_time_series, get_sliding_window_matrix


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


# Input files
# curr_state_folder = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 2'
# test_state_folder = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 3'
#
# all_state_folder = [
#     r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 2',
#     r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 3',
#     r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 4',
#     r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 6 fail motore',
# ]
#
# # Features
# features_list = [
#     "Acceleration_X1", "Acceleration_Y1", "Acceleration_Z1",
#     # "Acceleration_X2", "Acceleration_Y2", "Acceleration_Z2",
#     # "Acceleration_X3", "Acceleration_Y3", "Acceleration_Z3"
# ]
#
# # Data preparation params
# # kernel = 144  # 288
# # stride = 77   # 77
#
# kernel = 288
# stride = 77
#
# # start_sample = 500000
# # num_sample = float('inf')
# num_sample = 1000000
#
# train_step = 5
# test_step = 5
#
# # max_test_step = 5000
#
# # Isolation Forest params
# # model_type = 'setup_clustering'
# model_type = 'cnn'
# params_file = './params/params_{}.json'.format(model_type)
#
# save_result = True
# overwrite = True
# output_dir = './results'


def main():

    params = get_argument()
    all_state_folder = params['all_state_folder']
    features_list = params['features_list']
    kernel = params['kernel']
    stride = params['stride']
    model_type = params['model_type']
    train_step = params['train_step']
    test_step = params['test_step']

    num_sample = 1000000
    params_file = './params/params_{}.json'.format(model_type)
    save_result = True
    overwrite = True
    output_dir = './results'

    # train_state = os.path.basename(curr_state_folder)
    # test_state = os.path.basename(test_state_folder)
    # print('Current State folder: ', train_state)
    # print('Test State folder: ', test_state)
    #
    # if not os.path.isdir(curr_state_folder) or not os.path.isdir(test_state_folder):
    #     print('No folder selected')
    #     return

    # curr_files = get_files(curr_state_folder, ext='lvm')
    # test_files = get_files(test_state_folder, ext='lvm')

    result_array = []

    curr_files = []
    for folder in all_state_folder:
        curr_files += get_files(folder, ext='lvm')[:train_step]

    test_files = curr_files

    for train_file in curr_files[:]:
        print('\n' + '#' * 50)

        train_state = os.path.split(os.path.dirname(train_file))[-1]
        print("State Train: ", train_state)

        print("\nRead Train File: ", os.path.basename(train_file))
        ds_train = read_ds_lvm(train_file, get_header=False)

        # Check train
        if ds_train is None or ds_train.empty:
            print('Impossible read train file')
            continue

        train_len = len(ds_train)
        ds_train = ds_train[:min(train_len, num_sample)]
        print('Train Original File Length: ', train_len)
        print('New File Length {} {:.02f}'.format(len(ds_train), 100 * len(ds_train) / train_len))

        # Select features
        ds_train = ds_train[features_list]

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

        for test_file in test_files[:]:

            test_state = os.path.split(os.path.dirname(test_file))[-1]

            if train_state == test_state \
                    and test_file == train_file:
                continue

            print("\nState Test: ", test_state)
            print("Read Test File: ", os.path.basename(test_file))
            ds_test = read_ds_lvm(test_file, get_header=False)

            # Check test
            if ds_test is None or ds_test.empty:
                print('Impossible read test file')
                continue

            # Select features
            ds_test = ds_test[features_list]

            test_len = len(ds_test)
            ds_test = ds_test[:min(test_len, num_sample)]
            print('Test Original File Length: ', test_len)
            print('New File Length {} {:.02f}'.format(len(ds_test), 100 * len(ds_test) / test_len))

            # Testing
            print('Testing...')
            y_pred = predict_anomaly(ds_test, model, kernel, with_skip=True)

            # Encoding results into triplet formats
            results = create_triplet_time_series(y_pred, with_support=True)

            # Number of test samples of kernel length
            test_sample = len(ds_test) / kernel

            # Show results
            results = pd.DataFrame(results)
            if results.empty:
                tot = 0
                pct_tot = 0
                tot_sample = 0

                print("Results: NO Anomaly founded")
            else:
                tot = results['support'].sum()
                pct_tot = 100 * tot / len(ds_test)

                tot_sample = tot / kernel

                print("Results: {} (record {:.04f})".format(tot, pct_tot))
                print("Anomaly Sample: {} (test sample {:.04f})".format(int(tot_sample), test_sample))

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
                'NUM_SINGLE_ANOMALY': tot,
                'PCT_ANOMALY': pct_tot,
                'NUM_SAMPLE_ANOMALY': tot_sample,
                'NUM_SAMPLE': test_sample,
            }

            result_array.append(result_record)

    if save_result:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, 'results_' + model_type + '.csv')

        result_ds = pd.DataFrame(result_array)

        if os.path.isfile(filename) and not overwrite:
            prev_result_ds = pd.read_csv(filename)
            result_ds = pd.concat([prev_result_ds, result_ds], axis=0, ignore_index=True)

        result_ds.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
