"""
The implementation of LOF for unsupervised outlier detection.

Authors:
    scikit-learn Team

Reference:
    Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J.
    LOF: identifying density-based local outliers. In ACM sigmod record.

"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class LOF(object):

    def __init__(self, n_neighbors=20, algorithm='auto', metric='minkowski', contamination='auto', **kwargs):
        """ Local Outlier Factor
        Arguments
        ---------
        n_neighbors : int, default=20
            Number of neighbors to use by default for kneighbors queries.
            If n_neighbors is larger than the number of samples provided,
            all samples will be used.
        algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
            Algorithm used to compute the nearest neighbors.
        leaf_size : int, default=30
            Leaf size passed to BallTree or KDTree.
            This can affect the speed of the construction and query,
            as well as the memory required to store the tree.
            The optimal value depends on the nature of the problem
        metric : str or callable, default=’minkowski’
            metric used for the distance computation.
            Any metric from scikit-learn or scipy.spatial.distance can be used.
        contamination : ‘auto’ or float, default=’auto’
            The amount of contamination of the data set,
            i.e. the proportion of outliers in the data set.
            When fitting this is used to define the threshold on the scores of the samples.
        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        """
        self.model = LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors, algorithm=algorithm, metric=metric,
                                        contamination=contamination, **kwargs)

    def fit(self, X):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== LOF Fit ======')
        # X = X.reshape((len(X), -1))
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
        print('====== LOF Predict ======')
        # X = X.reshape((len(X), -1))
        y_pred = self.model.predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred
