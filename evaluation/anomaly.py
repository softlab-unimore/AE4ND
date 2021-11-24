import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid

from timely.utils.manage_file import get_files, read_ds_lvm
from timely.utils.tools import get_sliding_window_matrix
from timely.transforms.transformer import resample, get_transformer
from timely.models.anomaly_detection.cnn_autoencoder import CNNAutoEncoder
from timely.models.anomaly_detection.deep_autoencoder import DeepAutoEncoder
from timely.models.anomaly_detection.lstm_autoencoder import LSTMAutoEncoder
from timely.models.anomaly_detection.bilstm_autoencoder import BiLSTMAutoEncoder

from timely.models.anomaly_detection.pca import PCA
from timely.models.anomaly_detection.one_class_svm import OneClassSVM
from timely.models.anomaly_detection.setup_clustering import SetupClustering


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


def apply_transform(ds, transformer):
    return pd.DataFrame(transformer.transform(ds.values), columns=ds.columns)


def transform_data(ds_list, transform_type):
    # Concat list of dataframe
    ds_concat = pd.concat(ds_list, axis=0, ignore_index=True)

    # Train transformer
    transformer = get_transformer(ds_concat, transform_type)

    # Apply transformer
    ds_transformed_list = [apply_transform(ds, transformer) for ds in ds_list]

    return ds_transformed_list, transformer


def prepare_data(ds_list, labels, kernel, stride=1):
    # Create slide window matrix for each train
    x_list = [get_sliding_window_matrix(ds.values, kernel, stride) for ds in ds_list]

    # Assign labels for each matrix values
    y = np.hstack([[i] * len(x) for i, x in zip(labels, x_list)])

    # Concat each matrix
    x = np.vstack(x_list)

    return x, y


def get_deep_model(model_type, model_params=None):
    # Model initialization
    if model_params is None:
        model_params = {}

    print("Model initialization: ", model_type)
    if model_type == 'cnn':
        model = CNNAutoEncoder(**model_params)
    elif model_type == 'deep':
        model = DeepAutoEncoder(**model_params)
    elif model_type == 'lstm':
        model = LSTMAutoEncoder(**model_params)
    elif model_type == 'bilstm':
        model = BiLSTMAutoEncoder(**model_params)

    elif model_type == 'pca':
        model_params.pop('with_lazy', None)
        model = PCA(**model_params)
    elif model_type == 'svm':
        model_params.pop('with_lazy', None)
        model = OneClassSVM(**model_params)
    elif model_type == 'cluster':
        model_params.pop('with_lazy', None)
        model = SetupClustering(**model_params)

    else:
        raise ValueError('{} does not exist'.format(model_type))

    return model


def get_classification_report_record(report_dict):
    res = {}
    for k, val in report_dict.items():
        if isinstance(val, dict):
            for k_next, val_next in val.items():
                res['{}__{}'.format(k, k_next)] = val_next
        else:
            res[k] = val
    return res


def main():
    params = get_argument()
    all_state_folder = params['all_state_folder'][:3]
    features_list = params['features_list']
    model_type = params['model_type']
    resample_rate = 6400

    stride = 1
    epochs = 200

    save_result = True
    output_dir = './results'

    params_grid = {
        'kernel': [40, 80, 120, 200, 240, 360],
        'transform_type': ['minmax'],
        'with_lazy': [0.00],  # , 0.01, 0.015, 0.02],
        # 'loss': ['mae', 'mse'],
        # 'activation': [layers.LeakyReLU(alpha=0.3), 'relu', 'tanh']
    }
    # if model_type == 'bilstm':
    #     params_grid['activation'] = ['relu', 'tanh']

    # ['cluster', 'svm', 'pca', 'cnn', 'deep', 'lstm', 'bilstm']
    for model_type in ['pca', 'svm', 'cluster']:
        if model_type == 'pca':
            new_params = {
                'threshold': [1, 5, 10, 20, 50, 100],
            }
            params_grid.update(new_params)

        if model_type == 'svm':
            new_params = {
                'skernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'max_iter': [1000000],
                'gamma': ['scale', 'auto'],
            }
            params_grid.update(new_params)

        if model_type == 'cluster':
            new_params = {
                'max_distance': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1],
                'anomaly_threshold': [None]
            }
            params_grid.update(new_params)

        skip_list = [0]
        train_list = [1]

        combs = []
        states = list(range(len(all_state_folder)))
        for i in range(len(all_state_folder) - 1):
            r = i + 1
            l = list(itertools.combinations(states, r=r))
            l = [list(x) for x in l]

            combs += l

        for selected_states in combs:
            ds_train_list = []
            y_train_list = []
            ds_test_list = []
            y_test_list = []

            # Read train and test files
            print('Evaluation state: {}'.format(selected_states))
            for state_id, folder in enumerate(all_state_folder):
                print('Read state: ', os.path.basename(folder))
                files = get_files(folder, ext='lvm')
                for i, filename in enumerate(files):
                    if i in skip_list:
                        print('Skip: ', filename)
                        continue

                    ds = read_ds_lvm(filename, get_header=False)
                    ds = ds[features_list]
                    ds = resample(ds, resample_rate)

                    if i in train_list and state_id not in selected_states:
                        ds_train_list.append(ds)
                        print('Train state {} file: {}'.format(state_id, filename))
                        y_train_list.append(state_id)
                    else:
                        ds_test_list.append(ds)
                        print('Test state {} file: {}'.format(state_id, filename))
                        y_test_list.append(state_id)

            ds_res = []
            for grid in ParameterGrid(params_grid):

                print('\n Params:')
                print(grid)

                kernel = grid['kernel']
                transform_type = grid['transform_type']

                model_params = dict(grid)
                model_params.pop('kernel', None)
                model_params.pop('transform_type', None)
                if 'skernel' in model_params:
                    model_params['kernel'] = model_params['skernel']
                    model_params.pop('skernel', None)

                # Apply transform
                transformer = None
                if transform_type:
                    print('Apply transform: ', transform_type)
                    x_train_list, transformer = transform_data(ds_train_list, transform_type)
                    x_test_list = [apply_transform(ds, transformer) for ds in ds_test_list]

                else:
                    print('No transform selected')
                    x_train_list = ds_train_list
                    x_test_list = ds_test_list

                # Create train and test matrix set
                x_train, y_train = prepare_data(x_train_list, labels=y_train_list, kernel=kernel, stride=stride)
                x_test, y_test = prepare_data(x_test_list, labels=y_test_list, kernel=kernel, stride=stride)

                print('Train size:       ', x_train.shape)
                print('Train label size: ', y_train.shape)
                print('Test size:        ', x_test.shape)
                print('Test label size:  ', y_test.shape)

                order = np.random.permutation(len(x_train))
                x_new = x_train[order]
                y_new = y_train[order]

                # Model initialization
                print("Model initialization: {}".format(model_type))
                model = get_deep_model(model_type, model_params=model_params)

                # Training
                print("Training...")
                model.fit(x=x_new, epochs=epochs, verbose=2)

                if model_type in ['cnn', 'deep', 'lstm', 'bilstm']:
                    original_threshold = model.threshold
                    print('Anomaly threshold: ', original_threshold)

                    for th in [0.00, 0.01, 0.015, 0.02]:
                        model.threshold = original_threshold + th
                        print('\n Lazy ', model.threshold)

                        print("Anomaly accuracy")
                        y_pred = model.predict(x_test, classifier=False)
                        y_true = np.zeros(len(y_test))

                        for selected_state_id in selected_states:
                            y_true[y_test == selected_state_id] = 1

                        print(classification_report(y_true, y_pred))

                        report_dict = classification_report(y_true, y_pred, output_dict=True)
                        record = get_classification_report_record(report_dict)
                        record.update(grid)
                        record['with_lazy'] = th

                        ds_res.append(record)
                else:
                    print("Anomaly accuracy")
                    y_pred = model.predict(x_test)
                    y_true = np.zeros(len(y_test))

                    for selected_state_id in selected_states:
                        y_true[y_test == selected_state_id] = 1

                    print(classification_report(y_true, y_pred))

                    report_dict = classification_report(y_true, y_pred, output_dict=True)
                    record = get_classification_report_record(report_dict)
                    record.update(grid)
                    ds_res.append(record)

            ds_res = pd.DataFrame(ds_res)

            if save_result:
                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                name = [str(x) for x in selected_states]
                name = '_'.join(name)

                filename = os.path.join(output_dir, 'results_gimple_grid_anomaly__{}__{}.csv'.format(name, model_type))
                ds_res.to_csv(filename, index=True)


if __name__ == '__main__':
    main()
