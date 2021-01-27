"""
The implementation of CNN AutoEncoder models for anomaly detection.
"""
import numpy as np
# import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
# import matplotlib.pyplot as plt


class CNNAutoEncoder(object):

    def __init__(self, learning_rate=0.01):
        """ CNN AutoEncoder models for anomaly detection """
        self.sequence_length = None
        self.num_features = None

        self.learning_rate = learning_rate
        self.loss = 'mae'

        self.history = None
        self.threshold = 0

    def _set_input(self, X):
        assert len(X.shape) == 3, 'Invalid input shape'
        self.sequence_length = X.shape[1]
        self.num_features = X.shape[2]

    def _define_model(self, ):
        model = keras.Sequential(
            [
                layers.Input(shape=(self.sequence_length, self.num_features)),
                layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(filters=self.num_features, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss)
        # model.summary()

        self.model = model

    def _compute_reconstruction_error(self, x, x_pred):
        x = x.reshape((len(x), -1))
        x_pred = x_pred.reshape((len(x_pred), -1))
        mae_loss = np.mean(np.abs(x_pred - x), axis=1)
        return mae_loss

    def _set_reconstruction_error(self, x):
        # Get train MAE loss.
        x_pred = self.model.predict(x)

        train_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # plt.hist(train_mae_loss, bins=50)
        # plt.xlabel("Train MAE loss")
        # plt.ylabel("No of samples")
        # plt.show()

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        print("Reconstruction error threshold: ", threshold)
        self.threshold = np.max(threshold, self.threshold)

    def fit(self, x):
        print('CNN AutoEncoder Fit')
        self._set_input(x)
        self._define_model()

        history = self.model.fit(
            x=x, y=x,
            epochs=100,
            batch_size=128,
            validation_split=0.1,
            verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
            ],
        )
        self.history = history

        # plt.plot(history.history["loss"], label="Training Loss")
        # plt.plot(history.history["val_loss"], label="Validation Loss")
        # plt.legend()
        # plt.show()

        self._set_reconstruction_error(x)

    def tune(self, new_x):
        self.model.fit(
            x=new_x, y=new_x,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
            ],
        )

        self._set_reconstruction_error(new_x)

    def predict(self, x):
        print('CNN AutoEncoder Predict')
        x_pred = self.model.predict(x)

        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > self.threshold
        print("Number of anomaly samples: ", np.sum(anomalies))
        # print("Indices of anomaly samples: ", np.where(anomalies))

        return anomalies
