"""
The implementation of Deep AutoEncoder models for anomaly detection.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics


class DeepAutoEncoder(object):

    def __init__(self, activation='tanh', loss='mae', learning_rate=0.0004, with_lazy=0.02):
        """ Deep AutoEncoder models for anomaly detection """
        self.sequence_length = None
        self.num_features = None
        self.num_class = None

        # AutoEncoder alpha and loss
        self.learning_rate = learning_rate
        self.loss = loss

        # AutoEncoder activation function
        self.activation = activation

        # AutoEncoder and Classifier model
        self.model = None
        self.encoder = None
        self.classifier = None

        self.history = None
        self.history_classifier = None

        self.with_lazy = with_lazy
        self.threshold = 0

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
        x = layers.Dense(200, activation=self.activation)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(100, activation=self.activation)(x)
        encoded = layers.Dense(100, activation=self.activation)(x)

        encoder = keras.Model(input_series, encoded)

        # Deep decoder
        x = layers.Dense(100, activation=self.activation)(encoded)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(200, activation=self.activation)(x)
        x = layers.Dense(self.num_features * self.sequence_length)(x)
        decoded = layers.Reshape((self.sequence_length, self.num_features))(x)

        autoencoder = keras.Model(input_series, decoded)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0, clipvalue=0.5)
        autoencoder.compile(optimizer=optimizer, loss=self.loss)

        self.model = autoencoder
        self.encoder = encoder

        self.model.summary()

    def _define_classifier(self, ):
        # Classifier
        input_series = keras.Input(shape=(self.sequence_length, self.num_features))

        x = self.encoder(input_series)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(self.num_class, activation='softmax')(x)

        self.classifier = keras.Model(input_series, x)
        self.encoder.trainable = False

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = losses.SparseCategoricalCrossentropy()
        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=[metrics.SparseCategoricalAccuracy()])

        self.classifier.summary()

    def _compute_reconstruction_error(self, x, x_pred):
        # Compute mean average error for each dimension and for each sample
        x = x.reshape((len(x), -1))
        x_pred = x_pred.reshape((len(x_pred), -1))
        mae_loss = np.mean(np.abs(x_pred - x), axis=1)
        return mae_loss

    def _set_reconstruction_error(self, x):
        # Get train MAE loss.
        x_pred = self.model.predict(x)
        train_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        print("Reconstruction error threshold: ", threshold)
        print("Min error: ", np.min(train_mae_loss))
        print("Max error: ", np.max(train_mae_loss))
        print("Avg error: ", np.mean(train_mae_loss))
        print("Std error: ", np.std(train_mae_loss))

        if self.with_lazy:
            # threshold = threshold + np.std(train_mae_loss)
            # iqr = np.quantile(train_mae_loss, 0.75) - np.quantile(train_mae_loss, 0.25)
            # threshold = threshold + 10 * iqr
            # threshold = threshold + 0.0005
            threshold = threshold + self.with_lazy
            print("Use lazy reconstruction error threshold: ", threshold)

        self.threshold = np.max(threshold, self.threshold)

    def fit(self, x, y=None, epochs=100, verbose=0):
        print('Deep AutoEncoder Fit')
        # Define autoencoder input params
        self._set_input(x)

        # Construct the model for the given input
        self._define_model()

        # Train Deep AutoEncoder
        self.history = self.model.fit(
            x=x, y=x,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
            ],
        )

        # Save train reconstruction error
        self._set_reconstruction_error(x)

        # Classification task
        if y is not None:
            # Define classification params and model
            self._set_classes(y)
            self._define_classifier()

            # Train Classifier
            self.history_classifier = self.classifier.fit(
                x=x, y=y,
                epochs=epochs,
                batch_size=128,
                validation_split=0.1,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
                ],
            )

    def tune(self, new_x):
        pass

    def predict(self, x, classifier=False):
        print('Deep AutoEncoder Predict')
        if classifier:
            # Classifier prediction
            assert self.classifier is not None, 'Please train the classifier'

            # Detect sample class
            y_pred = self.classifier.predict(x)
            y_pred = y_pred.argmax(axis=1)
            return y_pred

        else:
            # AutoEncoder reconstruction
            x_pred = self.model.predict(x)

            # Reconstruction error
            test_mae_loss = self._compute_reconstruction_error(x, x_pred)

            # Detect all the samples which are anomalies
            anomalies = test_mae_loss > self.threshold

            print("Number of anomaly samples: ", np.sum(anomalies))
            print("Error mean: {} max: {}".format(np.mean(test_mae_loss), np.max(test_mae_loss)))

            return anomalies

    def decision_score(self, x):
        print('Deep AutoEncoder Decision Score')
        # AutoEncoder reconstruction
        x_pred = self.model.predict(x)

        # Reconstruction error
        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect the samples shift from the trained threshold
        scores = test_mae_loss - self.threshold

        return scores
