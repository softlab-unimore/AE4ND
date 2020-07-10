import logging
import numpy as np
from numpy import linalg as LA

import joblib
from pickle import UnpicklingError, PickleError

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, euclidean
from fastdtw import fastdtw


class SetupClustering(object):

    def __init__(self, max_dist=0.3, anomaly_threshold=0.3, distance='cosine',
                 mode='online',
                 num_bootstrap_samples=1000):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            mode: str, 'offline' or 'online' mode for clustering
            num_bootstrap_samples: int, online clustering starts with a bootstraping process, which
                determines the initial cluster representatives offline using a subset of samples
        """

        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.distance = self._get_distance_metric(distance)
        self.mode = mode
        self.num_bootstrap_samples = num_bootstrap_samples

        self.representatives = list()  # the representative samples of clusters, of shape num_clusters-by-num_events
        self.cluster_size_dict = dict()  # the size of each cluster, used to update representatives online

    @staticmethod
    def _get_distance_metric(distance):
        if distance == 'euclidean':
            return 'euclidean'
        elif distance == 'cosine':
            return 'cosine'
        elif distance == 'dtw':
            return 'dtw'
        else:
            raise ValueError('select the wrong distance metric')

    def fit(self, X):
        logging.info('====== SetupClustering Fit ======')
        X = X.reshape((len(X), -1))

        if self.mode == 'offline':
            # The offline mode can process about 10K samples only due to huge memory consumption.
            self._offline_clustering(X)

        elif self.mode == 'online':
            # Bootstrapping phase
            if self.num_bootstrap_samples > 0:
                X_bootstrap = X[0:self.num_bootstrap_samples, :]
                self._offline_clustering(X_bootstrap)

            # Online learning phase
            if X.shape[0] > self.num_bootstrap_samples:
                self._online_clustering(X)

    def predict(self, X):
        logging.info('====== SetupClustering Predict ======')
        X = X.reshape((len(X), -1))
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            min_dist, min_index = self._get_min_cluster_dist(X[i, :])
            if min_dist > self.anomaly_threshold:
                y_pred[i] = 1
        return y_pred

    def _offline_clustering(self, X):
        logging.info('Starting offline clustering...')
        p_dist = pdist(X, metric=self._distance_metric)
        z = linkage(p_dist, 'complete')
        cluster_index = fcluster(z, self.max_dist, criterion='distance')
        self._extract_representatives(X, cluster_index)
        logging.info('Processed {} instances.'.format(X.shape[0]))
        logging.info('Found {} clusters offline.\n'.format(len(self.representatives)))

    def _extract_representatives(self, X, cluster_index):
        num_clusters = len(set(cluster_index))
        for clu in range(num_clusters):
            clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
            self.cluster_size_dict[clu] = clu_idx.shape[0]
            representative_center = np.average(X[clu_idx, :], axis=0)
            self.representatives.append(representative_center)

    def _online_clustering(self, X):
        logging.info("Starting online clustering...")
        for i in range(self.num_bootstrap_samples, X.shape[0]):
            if (i + 1) % 2000 == 0:
                logging.info('Processed {} instances.'.format(i + 1))

            instance_vec = X[i, :]
            if len(self.representatives) > 0:
                min_dist, clu_id = self._get_min_cluster_dist(instance_vec)
                if min_dist <= self.max_dist:
                    self.cluster_size_dict[clu_id] += 1
                    self.representatives[clu_id] = self.representatives[clu_id] \
                                                   + (instance_vec - self.representatives[clu_id]) \
                                                   / self.cluster_size_dict[clu_id]
                    continue
            self.cluster_size_dict[len(self.representatives)] = 1
            self.representatives.append(instance_vec)
        logging.info('Processed {} instances.'.format(X.shape[0]))
        logging.info('Found {} clusters online.\n'.format(len(self.representatives)))

    def save_model(self, filename: str):
        try:
            joblib.dump(self.representatives, filename)
            return True
        except PickleError as e:
            logging.error(e)
            return False

    def load_model(self, filename: str):
        try:
            self.representatives = joblib.load(filename)
            assert isinstance(self.representatives, list)
            self.cluster_size_dict = {}
            for i, array in enumerate(self.representatives):
                assert isinstance(array, np.ndarray)
                self.cluster_size_dict[i] = 1
            return True
        except (UnpicklingError, AssertionError) as e:
            logging.error(e)
            return False

    @staticmethod
    def _cosine_metric(x1, x2):
        # x1 = x1.ravel()
        # x2 = x2.ravel()
        norm = LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        return distance

    @staticmethod
    def dtw(x1, x2):
        res, _ = fastdtw(x1, x2)
        return res

    def _distance_metric(self, x1, x2):
        if self.distance == 'euclidean':
            distance = euclidean(x1, x2)
        elif self.distance == 'cosine':
            distance = self._cosine_metric(x1, x2)
        elif self.distance == 'dtw':
            distance, _ = fastdtw(x1, x2)
        else:
            distance = 0
        return distance

    def _get_min_cluster_dist(self, instance_vec):
        min_index = -1
        min_dist = float('inf')
        for i in range(len(self.representatives)):
            cluster_rep = self.representatives[i]
            dist = self._distance_metric(instance_vec, cluster_rep)
            if dist < 1e-8:
                min_dist = 0
                min_index = i
                break
            elif dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index
