"""
The implementation of Deep AutoEncoder models for anomaly detection.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics


class DeepClassifier(object):

    def __init__(self, learning_rate=0.0004):
        """ Deep AutoEncoder models for anomaly detection """
        self.sequence_length = None
        self.num_features = None
        self.num_class = None

        # AutoEncoder alpha and loss
        self.learning_rate = learning_rate

        # AutoEncoder and Classifier model
        self.encoder = None
        self.classifier = None

        self.history_classifier = None

    def _set_input(self, x):
        assert len(x.shape) == 3, 'Invalid input shape'
        self.sequence_length = x.shape[1]
        self.num_features = x.shape[2]

    def _set_classes(self, y):
        assert len(y.shape) == 1, 'Invalid label shape'
        self.num_class = len(np.unique(y))

    def _define_model(self, ):
        input_series = keras.Input(shape=(self.sequence_length, self.num_features))

        # Deep encoder
        x = layers.Flatten()(input_series)
        x = layers.Dense(200, activation='tanh')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(100, activation='tanh')(x)
        encoded = layers.Dense(100, activation='tanh')(x)

        encoder = keras.Model(input_series, encoded)

        self.encoder = encoder

    def _define_classifier(self, ):
        # Classifier
        input_series = keras.Input(shape=(self.sequence_length, self.num_features))

        x = self.encoder(input_series)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self.num_class, activation='softmax')(x)

        self.classifier = keras.Model(input_series, x)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = losses.SparseCategoricalCrossentropy()
        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=[metrics.SparseCategoricalAccuracy()])

        self.classifier.summary()

    def fit(self, x, y, verbose=0, **kwargs):
        print('Deep AutoEncoder Fit')
        # Define autoencoder input params
        self._set_input(x)

        # Construct the model for the given input
        self._define_model()

        # Define classification params and model
        self._set_classes(y)
        self._define_classifier()

        # Train Classifier
        self.history_classifier = self.classifier.fit(
            x=x, y=y,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
            ],
        )

    def predict(self, x, **kwargs):
        print('Deep Predict')
        # Classifier prediction
        assert self.classifier is not None, 'Please train the classifier'

        # Detect sample class
        y_pred = self.classifier.predict(x)
        y_pred = y_pred.argmax(axis=1)
        return y_pred
