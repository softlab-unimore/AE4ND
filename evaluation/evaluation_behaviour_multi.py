import os
import json
import argparse
import numpy as np
import pandas as pd

from utils.manage_model import get_model, predict_anomaly
from utils.manage_file import get_files, read_ds_lvm
from utils.tools import create_triplet_time_series, get_sliding_window_matrix

from transforms.transformer import resample, resample_with_feature_extractor


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
    with_decision_score = params.get('with_decision_score', False)
    custom_resample = params.get('custom_resample', False)

    resample_rate = 6400  # 12800 sample are 1 second
    with_skip = False

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

    # Initialize result array to memorize result
    # for each train and test step
    result_array = []

    # Get files from selected folder to use for training and testing
    curr_files = []
    for folder in all_state_folder:
        curr_files += get_files(folder, ext='lvm')[:3]

    test_files = curr_files

    for train_file in curr_files:
        print('\n' + '#' * 70)

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
        if custom_resample:
            ds_train = resample_with_feature_extractor(ds_train, resample_rate)
        else:
            ds_train = resample(ds_train, resample_rate)
        # ds_train = ds_train[:num_sample]
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

        for test_file in test_files:

            test_state = os.path.split(os.path.dirname(test_file))[-1]

            if train_state == test_state \
                    and test_file == train_file:
                continue

            print("\n State Test: ", test_state)
            print("Read Test File: ", os.path.basename(test_file))
            ds_test = read_ds_lvm(test_file, get_header=False)

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
            }

            result_array.append(result_record)

            # # Testing
            # y_pred = predict_anomaly(ds_test, model, kernel, with_skip=with_skip)
            #
            # # Encoding results into triplet formats
            # results = create_triplet_time_series(y_pred, with_support=True)
            #
            # # Show results
            # results = pd.DataFrame(results)
            #
            # # Get test stride
            # test_stride = kernel if with_skip else 1
            # # Number of test samples of kernel length
            # test_sample = int((len(ds_test) - kernel) / test_stride) + 1
            #
            # if results.empty:
            #     tot, pct_tot, tot_sample = 0, 0, 0
            #     print("Results: NO Anomaly founded")
            # else:
            #     # Number of single anomaly point
            #     tot = results['support'].sum()
            #     pct_tot = 100 * tot / (test_sample * test_stride)
            #     print("Results: {} (record {:.02f})".format(tot, pct_tot))
            #
            #     # Number of anomaly sample
            #     tot_sample = int(tot / test_stride)
            #
            #     if with_skip:
            #         print("Anomaly Sample: {} (test sample {:.02f})".format(int(tot_sample), test_sample))
            #
            # result_record = {
            #     'MODEL': model_type,
            #     'KERNEL': kernel,
            #     'STRIDE': stride,
            #     'TRAIN_STATE': train_state,
            #     'TRAIN': os.path.basename(train_file),
            #     'TRAIN_SIZE': train_len,
            #     'TEST_STATE': test_state,
            #     'TEST': os.path.basename(test_file),
            #     'TEST_LEN': test_len,
            #     'NUM_SINGLE_ANOMALY': tot,
            #     'PCT_ANOMALY': pct_tot,
            #     'NUM_SAMPLE_ANOMALY': tot_sample,
            #     'NUM_SAMPLE': test_sample,
            # }
            #
            # result_array.append(result_record)

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
