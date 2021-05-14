"""
The implementation of One Class SVM models for unsupervised outlier detection.

Authors:
    scikit-learn Team

Reference:
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py
"""
import numpy as np
from sklearn.svm import OneClassSVM as SVM

from ...transforms.transformer import get_transformer


class OneClassSVM(object):

    def __init__(self, kernel='rbf', gamma='scale', tol=0.001, nu=0.5, shrinking=True, max_iter=1000):
        """
        Unsupervised Outlier Detection.
        Arguments
        ---------
            kernel : {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, optional (default=rbf).
                Specifies the kernel type to be used in the algorithm.
                It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable
            gamma : {‘scale’, ‘auto’} or float, default=’scale’
                Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            tol : float, default=1e-3
                Tolerance for stopping criterion
            nu : float, default=0.5
                An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
                Should be in the interval (0, 1]. By default 0.5 will be taken
            max_iter : int, default=-1
                Hard limit on iterations within solver, or -1 for no limit.
        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
        """
        self.model = SVM(kernel=kernel, gamma=gamma, tol=tol, nu=nu, shrinking=shrinking, max_iter=max_iter)
        self.transformer = None

    def fit(self, x):
        """
        Arguments
        ---------
            x: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('OneClassSVM Fit')
        x = x.reshape((len(x), -1))

        self.transformer = get_transformer(x, 'minmax')
        x = self.transformer.transform(x)

        self.model.fit(x)

    def predict(self, x):
        """ Predict anomalies with mined invariants
        Arguments
        ---------
            x: the input event count matrix
        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        print('OneClassSVM Predict')
        x = x.reshape((len(x), -1))
        x = self.transformer.transform(x)

        y_pred = self.model.predict(x)

        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred
