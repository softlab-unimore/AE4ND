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

    def __init__(self, n_neighbors=20, algorithm='auto', metric='minkowski'):
        """
        Local Outlier Factor
        Arguments
        ---------
        n_neighbors : int, default=20
            Number of neighbors to use by default for kneighbors queries.
            If n_neighbors is larger than the number of samples provided,
            all samples will be used.
        algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
            Algorithm used to compute the nearest neighbors.
        metric : str or callable, default=’minkowski’
            metric used for the distance computation.
            Any metric from scikit-learn or scipy.spatial.distance can be used.

        ---------
            For more information, please visit
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        """
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric,
                                        contamination=0.00001, novelty=True)

    def fit(self, x):
        """
        Arguments
        ---------
            x: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('LOF Fit')
        x = x.reshape((len(x), -1))
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
        print('LOF Predict')
        x = x.reshape((len(x), -1))
        y_pred = self.model.predict(x)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred
