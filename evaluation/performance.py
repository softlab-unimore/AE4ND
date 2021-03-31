import os
import numpy as np
import pandas as pd
from datetime import datetime

from utils.manage_model import get_model
from utils.manage_file import read_ds_lvm
from utils.tools import get_sliding_window_matrix

from transforms.transformer import resample


def main():
    output_dir = './results'

    selected_files = [
        "/export/static/pub/softlab/dataset_sbdio/Anomaly Detection/TEST 2/testaccelerometri.lvm",
        "/export/static/pub/softlab/dataset_sbdio/Anomaly Detection/TEST 2/testaccelerometri.lvm",
        "/export/static/pub/softlab/dataset_sbdio/Anomaly Detection/TEST 3/testaccelerometri_1.lvm",
        "/export/static/pub/softlab/dataset_sbdio/Anomaly Detection/TEST 4/testaccelerometri.lvm",
    ]

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

    # Initialize result array to memorize performance result
    result_array = []

    # Model cycle
    for model_type, kernel in zip(model_list, kernel_list):

        print('\n\n')
        print('\nModel: {}\n'.format(model_type))

        params_file = './params/params_{}.json'.format(model_type)

        # Train cycle
        for i in range(len(selected_files)):
            x_train = []

            # Get train
            for pos, train_file in enumerate(selected_files[:i + 1]):
                if i > 0 and pos == 0:
                    continue

                ds_train = read_ds_lvm(train_file, get_header=False)

                if ds_train is None or ds_train.empty:
                    raise ValueError('Impossible read train file')

                ds_train = ds_train[features_list]
                ds_train = resample(ds_train, resample_rate)
                x = get_sliding_window_matrix(ds_train.values, kernel, stride)

                if pos == 0:
                    x = x[:len(x) // 2]

                x_train.append(x)

            # Train set
            x_train = np.vstack(x_train)

            print('\nTrain size: {}\n'.format(len(x_train)))

            # Model init
            model = get_model(model_type, params_file=params_file)

            # Model training
            train_start = datetime.now()
            model.fit(x_train)
            train_end = datetime.now()

            # Test cycle
            for j in range(len(selected_files)):

                x_test = []

                # Get test
                for pos, test_file in enumerate(selected_files[:j + 1]):
                    if j > 0 and pos == 0:
                        continue

                    ds_test = read_ds_lvm(test_file, get_header=False)

                    if ds_test is None or ds_test.empty:
                        raise ValueError('Impossible read test file')

                    ds_test = ds_test[features_list]
                    ds_test = resample(ds_test, resample_rate)
                    x = get_sliding_window_matrix(ds_test.values, kernel, stride)

                    if pos == 0:
                        x = x[:1]

                    x_test.append(x)

                # Test set
                x_test = np.vstack(x_test)

                print('\nTest size: {}\n'.format(len(x_test)))

                # Model predict
                test_start = datetime.now()
                model.predict(x_test)
                test_end = datetime.now()

                result_record = {
                    'model': model_type,
                    'train_size': len(x_train),
                    'train_time': train_end - train_start,
                    'test_size': len(x_test),
                    'test_time': test_end - test_start,
                }

                result_array.append(result_record)

    # Save results
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'performance.csv')
    result_ds = pd.DataFrame(result_array)
    result_ds.to_csv(filename, index=False)


if __name__ == '__main__':
    main()
