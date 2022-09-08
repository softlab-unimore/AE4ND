import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, adjusted_mutual_info_score

from ae4nd.utils.tools import prepare_data
from ae4nd.transformations.transformer import fit_and_transform_data_list, transform_data_list
from ae4nd.models.autoencoder import AutoEncoder


def main():
    window = 200  # Windows size to sample the multivariate time series during the sliding window process
    stride = 1  # How far the window should move at each step,
    transform_type = 'minmax'  # Type of transformation to apply to all the selected time series
    model_type = 'cnn'  # Autoencoder models: cnn, fcnn, lstm, bilstm

    # List of files used to train the autoencoder to learn normal behaviors
    train_files = [('data/series_state_1_1.csv', 1), ('data/series_state_2_1.csv', 2), ]

    # List of files used to test the trained autoencoder to be able to detect normal and novel behaviors
    test_files = [('data/series_state_1_2.csv', 1), ('data/series_state_2_2.csv', 2),
                  ('data/series_state_3_1.csv', 3), ]

    # Read train and test files
    df_train_list = [pd.read_csv(file) for file, _ in train_files]
    y_train_list = [label for _, label in train_files]
    df_test_list = [pd.read_csv(file) for file, _ in test_files]
    y_test_list = [label for _, label in test_files]

    # Apply the selected transformation
    x_train_list, transformer = fit_and_transform_data_list(df_train_list, transform_type)
    x_test_list = transform_data_list(df_test_list, transformer)

    # Create train and test matrix set
    x_train, y_train = prepare_data(x_train_list, labels=y_train_list, window=window, stride=stride)
    x_test, y_test = prepare_data(x_test_list, labels=y_test_list, window=window, stride=stride)

    # Randomize training data
    order = np.random.permutation(len(x_train))
    x_train = x_train[order]
    y_train = y_train[order]

    # AutoEncoder unsupervised novelty detection
    model = AutoEncoder(model_type=model_type)  # define the autoencoder
    model.fit(x_train, epochs=10, batch_size=32, verbose=0)  # train the autoencoder

    # Predict novel and normal states for each sample
    y_pred = model.predict(x_test)

    # Compute labels for novelty detection task
    print('\nNovelty detection accuracy')
    y_true = [1 if label not in y_train_list else 0 for label in y_test]
    print(classification_report(y_true, y_pred))

    # Classify record detected like normal state
    x_test2 = x_test[y_pred == 0]
    y_test2 = y_test[y_pred == 0]
    # Autoencoder unsupervised classification
    y_pred2 = model.classify(x_test2, supervised=False, n_clusters=2)
    print('\nUnsupervised classification accuracy')
    print('AMI', adjusted_mutual_info_score(y_test2, y_pred2))


if __name__ == '__main__':
    main()
