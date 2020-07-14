"""
The implementation of IsolationForest models for anomaly detection.

Authors:
    scikit-learn Team

Reference:
    Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua.
    “Isolation forest.” Data Mining, 2008. ICDM’08.

    Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua.
    “Isolation-based anomaly detection.” ACM Transactions on Knowledge Discovery from Data (TKDD)
    """
import logging
import numpy as np
from sklearn.ensemble import IsolationForest as iForest


class IsolationForest(object):

    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.03, max_features=0.5, **kwargs):
        """ The IsolationForest models for anomaly detection
        Arguments
        ---------
            n_estimators : int, optional (default=100). The number of base estimators in the ensemble.
            max_samples : int or float, optional (default="auto")
                The number of samples to draw from X to train each base estimator.
                    - If int, then draw max_samples samples.
                    - If float, then draw max_samples * X.shape[0] samples.
                    - If "auto", then max_samples=min(256, n_samples).
                If max_samples is larger than the number of samples provided, all samples will be used
                for all trees (no sampling).
            contamination : float in (0., 0.5), optional (default='auto')
                The amount of contamination of the data set, i.e. the proportion of outliers in the data
                set. Used when fitting to define the threshold on the decision function. If 'auto', the
                decision function threshold is determined as in the original paper.
            max_features : int or float, optional (default=1.0)
                The number of features to draw from X to train each base estimator.
                    - If int, then draw max_features features.
                    - If float, then draw max_features * X.shape[1] features.
            bootstrap : boolean, optional (default=False)
                If True, individual trees are fit on random subsets of the training data sampled with replacement.
                If False, sampling without replacement is performed.
            n_jobs : int or None, optional (default=None)
                The number of jobs to run in parallel for both fit and predict. None means 1 unless in a
                joblib.parallel_backend context. -1 means using all processors.
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used by np.random.

        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        """
        self.model = iForest(n_estimators=n_estimators,
                             max_samples=max_samples,
                             contamination=contamination,
                             max_features=max_features, **kwargs)

    def fit(self, X):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        logging.info('====== Isolation Forest Fit ======')
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
        logging.info('====== Isolation Forest Predict ======')
        X = X.reshape((len(X), -1))
        y_pred = self.model.predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred
