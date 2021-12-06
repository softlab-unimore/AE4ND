"""
The implementation of Deep AutoEncoder models for anomaly detection.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class LinearClassifier(object):

    def __init__(self, max_iter=10000, verbose=0):
        """ LinearClassifier models for state classification """
        # Data params
        self.sequence_length = None
        self.num_features = None
        self.num_class = None

        # Linear classifier model
        self.model = None

        # Model params
        self.max_iter = max_iter

        # Logging params
        self.verbose = verbose

    def _set_input(self, x):
        assert len(x.shape) == 3, 'Invalid input shape'
        self.sequence_length = x.shape[1]
        self.num_features = x.shape[2]

    def _set_classes(self, y):
        assert len(y.shape) == 1, 'Invalid label shape'
        self.num_class = len(np.unique(y))

    def _define_model(self, ):
        # self.model = LogisticRegression(
        #     max_iter=self.max_iter,
        #     verbose=self.verbose,
        # )

        self.model = LogisticRegression()

    def fit(self, x, y, **kwargs):
        print('Linear Fit')
        # Define autoencoder input params
        self._set_input(x)

        # Define classification params and model
        self._set_classes(y)

        # Construct the model for the given input
        self._define_model()

        # Train Classifier
        self.model.fit(x.reshape(len(x), -1), y)

    def predict(self, x, **kwargs):
        print('Linear Predict')
        # Classifier prediction
        assert self.model is not None, 'Please train the classifier'

        # Detect sample class
        y_pred = self.model.predict(x.reshape(len(x), -1))
        return y_pred
