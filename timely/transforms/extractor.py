import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis as kurt


def mean(arr):
    return np.mean(arr, axis=0)


def peak(arr):
    return np.max(arr, axis=0)


def peak_to_peak(arr):
    return np.max(arr, axis=0) - np.min(arr, axis=0)


def rms(arr):
    return np.sqrt(np.sum(np.power(arr, 2), axis=0) * (1 / len(arr)))


def crest_factor(arr):
    return peak(arr) / mean(arr)


def skewness(arr):
    return skew(arr, axis=0)


def kurtosis(arr):
    return kurt(arr, axis=0)


def shape_factor(arr):
    return rms(arr) / mean(arr)

# def impulsive_factor(arr):
#     return
