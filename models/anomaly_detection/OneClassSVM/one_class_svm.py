"""
The implementation of One Class SVM models for unsupervised outlier detection.

Authors:
    scikit-learn Team

Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
"""
import logging
import numpy as np
from sklearn.svm import OneClassSVM as SVM


class OneClassSVM(object):

    def __init__(self, kernel='rbf', gamma='scale', tol=0.001, nu=0.5, shrinking=True, **kwargs):
        """ Unsupervised Outlier Detection.
        Arguments
        ---------
            kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, optional (default=rbf).
                Specifies the kernel type to be used in the algorithm.
                It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
            degree : int, default=3
                Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
            gamma : {‘scale’, ‘auto’} or float, default=’scale’
                Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            coef0 : float, default=0.0
                Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
            tol : float, default=1e-3
                Tolerance for stopping criterion
            nu : float, default=0.5
                An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
                Should be in the interval (0, 1]. By default 0.5 will be taken
            shrinking : bool, default=True
                Whether to use the shrinking heuristic
            cache_size : float, default=200
                Specify the size of the kernel cache (in MB).
        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
        """
        self.model = SVM(kernel=kernel, gamma=gamma, tol=tol, nu=nu, shrinking=shrinking, **kwargs)

    def fit(self, X):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        logging.info('====== OneClassSVM Fit ======')
        X = X.reshape((len(X), -1))
        self.model.fit(X)

    def predict(self, X):
        """ Predict anomalies with mined invariants
        Arguments
        ---------
            X: the input event count matrix
        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        logging.info('====== OneClassSVM Predict ======')
        X = X.reshape((len(X), -1))
        y_pred = self.model.predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred
