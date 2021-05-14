"""
The implementation of CNN AutoEncoder models for anomaly detection.
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from ...transforms.transformer import get_transformer


class DeepAutoEncoder(object):

    def __init__(self, with_lazy=0.5, learning_rate=0.0004):
        """ Deep AutoEncoder models for anomaly detection """
        self.sequence_length = None
        self.num_features = None

        self.learning_rate = learning_rate
        self.loss = 'mae'

        self.transformer = None

        self.history = None
        self.with_lazy = with_lazy
        self.threshold = 0

    def _set_input(self, X):
        assert len(X.shape) == 2, 'Invalid input shape'
        self.num_features = X.shape[1]

    def _define_model(self, ):
        model = keras.Sequential(
            [
                layers.Input(shape=self.num_features),
                layers.Dense(96, activation='tanh'),
                layers.Dropout(rate=0.2),
                layers.Dense(64, activation='tanh'),
                layers.Dense(32, activation='tanh'),
                layers.Dense(64, activation='tanh'),
                layers.Dropout(rate=0.2),
                layers.Dense(96, activation='tanh'),
                layers.Dense(self.num_features),
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

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        print("Reconstruction error threshold: ", threshold)
        print("Min error: ", np.min(train_mae_loss))
        print("Max error: ", np.max(train_mae_loss))
        print("Average error: ", np.mean(train_mae_loss))
        print("Std error: ", np.std(train_mae_loss))

        if self.with_lazy > 0:
            threshold = threshold + self.with_lazy
            print("Use lazy reconstruction error threshold: ", threshold)

        self.threshold = np.max(threshold, self.threshold)

    def fit(self, x):
        print('Deep AutoEncoder Fit')
        x = x.reshape((len(x), -1))
        self._set_input(x)
        self._define_model()

        self.transformer = get_transformer(x, 'std')
        x = self.transformer.transform(x)

        history = self.model.fit(
            x=x, y=x,
            epochs=200,
            batch_size=128,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")
            ],
        )
        self.history = history

        self._set_reconstruction_error(x)

    def tune(self, new_x):
        new_x = new_x.reshape((len(new_x), -1))

        new_x = self.transformer.transform(new_x)

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
        print('Deep AutoEncoder Predict')
        x = x.reshape((len(x), -1))

        x = self.transformer.transform(x)

        x_pred = self.model.predict(x)

        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > self.threshold
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Mean reconstruction error: ", np.mean(test_mae_loss))
        print("Max reconstruction error: ", np.max(test_mae_loss))

        # print("Indices of anomaly samples: ", np.where(anomalies))

        return anomalies

    def decision_score(self, x):
        print('Deep AutoEncoder Decision Score')
        x = x.reshape((len(x), -1))

        x = self.transformer.transform(x)

        x_pred = self.model.predict(x)

        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect all the samples which are anomalies.
        scores = test_mae_loss - self.threshold
        print("Number of anomaly samples: ", np.sum(scores > 0))

        print("Mean reconstruction error: {:.05f}".format(np.mean(test_mae_loss)))
        print("Mean distance from threshold: {:.05f}".format(np.mean(scores)))

        return scores
