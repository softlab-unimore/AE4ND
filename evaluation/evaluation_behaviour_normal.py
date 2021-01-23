import os
import pandas as pd
from tabulate import tabulate

from utils.manage_model import get_model, predict_anomaly
from utils.manage_file import get_files, read_ds_lvm
from utils.tools import create_triplet_time_series, get_sliding_window_matrix

# Input files
curr_state_folder = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 2'
test_state_folder = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 3'

# Features
features_list = [
    "Acceleration_X1", "Acceleration_Y1", "Acceleration_Z1",
    # "Acceleration_X2", "Acceleration_Y2", "Acceleration_Z2",
    # "Acceleration_X3", "Acceleration_Y3", "Acceleration_Z3"
]

# Data preparation params
# kernel = 120
# stride = 120

kernel = 120
stride = 120

num_sample = 500000
# num_sample = float('inf')

# max_test_step = 5000

# Isolation Forest params
# model_type = 'setup_clustering'
model_type = 'isolation_forest'
params_file = './params/params_{}.json'.format(model_type)

save_result = False
output_dir = './results'


def main():
    print('Current State folder: ', os.path.basename(curr_state_folder))
    print('Test State folder: ', os.path.basename(test_state_folder))

    if not os.path.isdir(curr_state_folder) or not os.path.isdir(test_state_folder):
        print('No folder selected')
        return

    curr_files = get_files(curr_state_folder, ext='lvm')
    test_files = get_files(test_state_folder, ext='lvm')

    for train_file in curr_files[:2]:
        print('\n' + '#' * 50)
        print("\nRead Train File: ", os.path.basename(train_file))
        ds_train = read_ds_lvm(train_file, get_header=False)

        # Check train
        if ds_train is None or ds_train.empty:
            print('Impossible read train file')
            continue

        print('Train Original File Length: ', len(ds_train))
        start_len = len(ds_train)
        ds_train = ds_train[:min(start_len, num_sample)]
        print('New File Length {} {:.02f}'.format(len(ds_train), 100 * len(ds_train) / start_len))

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

        for test_file in test_files[:5]:
            if curr_state_folder == test_state_folder \
                    and test_file == train_file:
                continue

            print("\nRead Test File: ", os.path.basename(test_file))
            ds_test = read_ds_lvm(test_file, get_header=False)

            # Check test
            if ds_test is None or ds_test.empty:
                print('Impossible read test file')
                continue

            # Select features
            ds_test = ds_test[features_list]

            print('Test Original File Length: ', len(ds_test))
            start_len = len(ds_test)
            ds_test = ds_test[:min(start_len, num_sample)]
            print('New File Length {} {:.02f}'.format(len(ds_test), 100 * len(ds_test) / start_len))

            # Testing
            print('Testing...')
            y_pred = predict_anomaly(ds_test, model, kernel, with_skip=True)

            # Encoding results into triplet formats
            results = create_triplet_time_series(y_pred, with_support=True)

            # Show results
            results = pd.DataFrame(results)
            if results.empty:
                print("Results: NO Anomaly founded")
            else:
                tot = results['support'].sum()
                pct_tot = tot / len(ds_test)

                test_sample = len(ds_test) / kernel
                tot_sample = tot / kernel

                print("Results: {} (record {:.04f})".format(tot, pct_tot))
                print("Anomaly Sample: {} (test sample {:.04f})".format(int(tot_sample), test_sample))


if __name__ == '__main__':
    main()
