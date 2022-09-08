import numpy as np


def get_sliding_window_matrix(data: np.array, window: int, stride: int):
    """convert input matrix using sliding window"""
    rows, cols = data.shape
    new_rows = 1 + (rows - window) // stride

    matrix = np.zeros((new_rows, window, cols))

    for i in range(new_rows):
        left = i * stride
        right = left + window
        matrix[i, :, :] = data[left:right, :]

    return matrix


def prepare_data(df_list: list, labels: list, window: int, stride: int = 1):
    """ Apply a sliding window process on a list of multivariate time series """
    # Create sliding window matrix for each train
    x_list = [get_sliding_window_matrix(df.values, window, stride) for df in df_list]

    # Assign labels for each matrix values
    y = np.hstack([[i] * len(x) for i, x in zip(labels, x_list)])

    # Concat each matrix
    x = np.vstack(x_list)

    return x, y
