import os
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from model.anomaly_detection.LOF.local_outlier_factor import LOF
from utils.tools import create_triplet_time_series, get_time_series_dataset, get_sliding_window_matrix

# Input files
train_file = './data/ts_normal1.CSV'
test_file = './data/ts_anomaly_setup3.CSV'
datetime_col = 'DT'
sep = ';'

# Empty list will select all input columns like features
features_list = [
    # '%E1',
    # 'TEMP_E1',
    # '%E2',
    'TEMP_E2',
    # '%E3',
    'TEMP_E3',
    # '%E4',
    'TEMP_E4',
    # '%E5',
    'TEMP_E5',
]

# Data preparation params
kernel = 200
stride = 10

# LOF params
n_neighbors = 50
algorithm = 'auto'
metric = 'minkowski'
contamination = 'auto'

# Demo params
save = True
save_model = False
visualize = False
output_dir = './results'


def main():
    print('Read input data')

    # Get train dataset
    print('train: {}'.format(train_file))
    ds_train = get_time_series_dataset(filename=train_file, sep=sep, col=datetime_col)
    # Check train
    if ds_train is None:
        raise ValueError('Impossible read train file')

    # Get test dataset
    print('test: {}'.format(test_file))
    ds_test = get_time_series_dataset(filename=test_file, sep=sep, col=datetime_col)
    # Check test
    if ds_test is None:
        raise ValueError('Impossible read test file')
    print('from {} to {}'.format(ds_test.index.min(), ds_test.index.max()))

    # Get features
    print('Select features')
    features = features_list
    if not features:
        features = ds_train.columns.to_list()
    elif set(features).difference(set(ds_train.columns)):
        raise ValueError('Select the wrong features')

    # Select features
    ds_train = ds_train[features]
    ds_test = ds_test[features]

    if visualize:
        fig, ax = plt.subplots(figsize=(20, 10))
        ds_train.plot(ax=ax)
        ax.set_title('Training Data')
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 10))
        ds_test.plot(ax=ax)
        ax.set_title('Test Data')
        plt.show()

    # Model initialization
    print("Model initialization")
    model = LOF(n_neighbors=n_neighbors,
                algorithm=algorithm,
                metric=metric,
                contamination=contamination,
                n_jobs=-1
                )

    # Create training set
    print("Create training set")
    x_train = get_sliding_window_matrix(ds_train.values, kernel, stride)

    # Training
    print("Training...")
    model.fit(x_train)

    # Option 1: Save trained model
    if save_model:
        # Create output directory
        filename = os.path.join(output_dir, 'model.pkl')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save trained model
        joblib.dump(model, filename)

        # Load trained model
        model = joblib.load(filename)

    # Create test set
    print("Create testing set")
    x_test = get_sliding_window_matrix(ds_test.values, kernel, kernel)

    # Testing
    print('Testing...')
    y_pred = model.predict(x_test)

    # Expand results
    y_pred = [val for val in y_pred for _ in range(kernel)]
    res = np.zeros((len(ds_test)))
    res[:len(y_pred)] = y_pred
    y_pred = pd.Series(res, index=ds_test.index, name='features')

    # Encoding results into triplet formats
    results = create_triplet_time_series(y_pred)

    # Show results
    print("Results:")
    results = pd.DataFrame(results)
    print(tabulate(results, headers='keys', tablefmt='psql'))

    # Save results
    if save:
        filename = os.path.basename(test_file)
        filename = os.path.join(output_dir, 'lof_' + filename)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results.to_csv(filename, sep=sep, index=False)


if __name__ == '__main__':
    main()
