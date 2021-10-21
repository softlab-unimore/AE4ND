import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from timely.utils.manage_model import get_model
from timely.utils.manage_file import get_files, read_ds_lvm
from timely.utils.tools import get_sliding_window_matrix
from timely.transforms.transformer import resample, resample_with_feature_extractor, get_transformer


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


def main():
    params = get_argument()
    all_state_folder = params['all_state_folder']
    features_list = params['features_list']
    resample_rate = 6400

    stride = 1
    epochs = 500

    transform_type = None # 'minmax'

    save_result = True
    output_dir = './results'

    for train_id in [1, 2]:
        skip_list = []
        train_list = [train_id]
        ds_train_list = []
        y_train_list = []
        ds_test_list = []
        y_test_list = []

        # Read train and test files
        print('Read all datasets')
        for state_id, folder in enumerate(all_state_folder):
            print('\nRead state: ', os.path.basename(folder))
            files = get_files(folder, ext='lvm')

            selected_train_id = [x for x in range(len(files)) if x in train_list]
            if not len(selected_train_id):
                selected_train_id = [1]

            for i, filename in enumerate(files):
                if i in skip_list:
                    print('Skip:               {}'.format(filename))
                    continue

                # ds = None
                ds = read_ds_lvm(filename, get_header=False)
                ds = ds[features_list]
                ds = resample(ds, resample_rate)

                if i in selected_train_id:
                    print('Train state {} file: {}'.format(state_id, filename))
                    ds_train_list.append(ds)
                    y_train_list.append(state_id)
                else:
                    print('Test state {} file:  {}'.format(state_id, filename))
                    ds_test_list.append(ds)
                    y_test_list.append(state_id)

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

        for kernel in [40, 80, 120, 200, 240, 360]:

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

            for model_type in ['linear', 'classifier', 'cnn', 'deep', 'lstm', 'bilstm']:

                # Model initialization
                print("Model initialization: {}".format(model_type))
                model = get_model(model_type)

                # Training
                print("Training...")
                model.fit(x=x_new, y=y_new, epochs=epochs, verbose=2)

                y_pred = model.predict(x_test, classifier=True)

                print(classification_report(y_test, y_pred))
                ds_res = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

                if save_result:
                    if not os.path.isdir(output_dir):
                        os.makedirs(output_dir, exist_ok=True)

                    filename = os.path.join(output_dir, 'results_{}_accuracy_{}_{}.csv'.format(train_id, model_type, kernel))
                    ds_res.to_csv(filename, index=True)


if __name__ == '__main__':
    main()
