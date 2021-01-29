import os
import numpy as np
import pandas as pd

from utils.manage_model import get_model, predict_anomaly
from utils.manage_file import read_ds_lvm
from utils.tools import create_triplet_time_series, get_sliding_window_matrix

from transforms.transformer import resample, resample_with_feature_extractor

# Input files
# train_file = r'data/simulation/testaccelerometri_2.lvm'
# test_file = r'data/simulation/testaccelerometri_3.lvm'

train_file = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 2\testaccelerometri(2).lvm'
test_file = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 2\testaccelerometri(1).lvm'
# test_file = r'C:\Users\delbu\Projects\Dataset\Anomaly Detection\TEST 3\testaccelerometri_1(2).lvm'

# Features
features_list = [
    "Acceleration_X1", "Acceleration_Y1", "Acceleration_Z1",
    "Acceleration_X2", "Acceleration_Y2", "Acceleration_Z2",
    "Acceleration_X3", "Acceleration_Y3", "Acceleration_Z3"
]

# Data preparation params
kernel = 120
stride = 1

resample_rate = 6400  # 12800 sample are 1 second
custom_resample = False

# Model params
model_type = 'setup_clustering'
params_file = './params/params_{}.json'.format(model_type)

# Type of test
with_skip = False
with_decision_score = False


def main():
    train_state = os.path.split(os.path.dirname(train_file))[-1]
    print("\n State Train: ", train_state)
    print("Read Train File: ", os.path.basename(train_file))
    ds_train = read_ds_lvm(train_file, get_header=False)

    # Check train
    if ds_train is None or ds_train.empty:
        print('Impossible read train file')
        return

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

    test_state = os.path.split(os.path.dirname(test_file))[-1]
    print("\n State Test: ", test_state)
    print("Read Test File: ", os.path.basename(test_file))
    ds_test = read_ds_lvm(test_file, get_header=False)

    # Check test
    if ds_test is None or ds_test.empty:
        print('Impossible read test file')
        return

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

    # Testing
    # y_pred = predict_anomaly(ds_test, model, kernel, with_skip=with_skip)

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
              "({:.05f} {:.05f} total {})".format(
            num_error, mean_error, mean_only_error, len(x_test)))

    # Encoding results into triplet formats
    # results = create_triplet_time_series(y_pred, with_support=True)
    #
    # # Show results
    # results = pd.DataFrame(results)
    # if results.empty:
    #     print("Results: NO Anomaly founded")
    # else:
    #     # print(tabulate(results, headers='keys', tablefmt='psql'))
    #
    #     test_stride = kernel if with_skip else 1
    #
    #     # Number of test samples of kernel length
    #     test_sample = int((len(ds_test) - kernel) / test_stride) + 1
    #
    #     # Number of single anomaly point
    #     tot = results['support'].sum()
    #     pct_tot = 100 * tot / (test_sample * test_stride)
    #
    #     print("Results: {} (record {:.02f})".format(tot, pct_tot))
    #
    #     if with_skip:
    #         # Number of anomaly sample
    #         tot_sample = int(tot / test_stride)
    #         print("Anomaly Sample: {} (test sample {:.02f})".format(int(tot_sample), test_sample))


if __name__ == '__main__':
    main()
