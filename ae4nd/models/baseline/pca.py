"""
The implementation of PCA models for anomaly detection.
Authors:
    LogPAI Team
Reference:
    Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan.
    Large-Scale System Problems Detection by Mining Console Logs. SOSP 2009.

    https://github.com/logpai/loglizer
"""

import numpy as np


class PCA(object):

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        """
        The PCA models for anomaly detection
        Arguments
        ----------
            n_components: float/int, number of principal compnents or the variance ratio they cover
            threshold: float, the anomaly detection threshold. When setting to None, the threshold
                is automatically calculated using Q-statistics
            c_alpha: float, the c_alpha parameter for caculating anomaly detection threshold using
                Q-statistics. The following is lookup table for c_alpha:
                c_alpha = 1.7507; # alpha = 0.08
                c_alpha = 1.9600; # alpha = 0.05
                c_alpha = 2.5758; # alpha = 0.01
                c_alpha = 2.807; # alpha = 0.005
                c_alpha = 2.9677;  # alpha = 0.003
                c_alpha = 3.2905;  # alpha = 0.001
                c_alpha = 3.4808;  # alpha = 0.0005
                c_alpha = 3.8906;  # alpha = 0.0001
                c_alpha = 4.4172;  # alpha = 0.00001
        """

        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha

    def _compute_simple_threshold(self, x):
        X = x.reshape((len(x), -1))
        assert self.proj_C is not None  # PCA models needs to be trained before prediction.
        errors = []
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            errors.append(SPE)

        self.threshold = np.max(errors)

    def fit(self, x, **kwargs):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('PCA Fit ')
        x = x.reshape((len(x), -1))
        num_instances, num_events = x.shape
        x_cov = np.dot(x.T, x) / float(num_instances)
        U, sigma, V = np.linalg.svd(x_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        print('n_components: {}'.format(n_components))
        print('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            # # Calculate threshold using Q-statistic. Information can be found at:
            # # http://conferences.sigcomm.org/sigcomm/2004/papers/p405-lakhina111.pdf
            # phi = np.zeros(3)
            # for i in range(3):
            #     for j in range(n_components, num_events):
            #         phi[i] += np.power(sigma[j], i + 1)
            # h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            # self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
            #                                    + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]),
            #                                    1.0 / h0)

            self._compute_simple_threshold(x)

        print('SPE threshold: {}\n'.format(self.threshold))

    def predict(self, x, **kwargs):
        print('PCA Predict')
        X = x.reshape((len(x), -1))
        assert self.proj_C is not None  # PCA models needs to be trained before prediction.
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            if SPE > self.threshold:
                y_pred[i] = 1
        return y_pred
