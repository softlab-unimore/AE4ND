import numpy as np
from tensorflow import keras
from sklearn.cluster import AgglomerativeClustering


def define_lstm_autoencoder(sequence_length: int, num_features: int, activation: str = 'tanh'):
    # Model input
    input_series = keras.Input(shape=(sequence_length, num_features))

    # LSTM encoder
    x = keras.layers.LSTM(200, activation=activation, return_sequences=True)(input_series)
    encoded = keras.layers.LSTM(100, activation=activation, dropout=0.2, return_sequences=False)(x)
    encoder = keras.Model(input_series, encoded)

    # LSTM decoder
    x = keras.layers.RepeatVector(sequence_length)(encoded)
    x = keras.layers.LSTM(100, activation=activation, return_sequences=True)(x)
    x = keras.layers.LSTM(200, activation=activation, return_sequences=True)(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(16))(x)
    decoded = keras.layers.TimeDistributed(keras.layers.Dense(num_features))(x)

    # Autoencoder
    autoencoder = keras.Model(input_series, decoded)

    return encoder, autoencoder


def define_bilstm_autoencoder(sequence_length: int, num_features: int, activation: str = 'tanh'):
    # Model input
    input_series = keras.Input(shape=(sequence_length, num_features))

    # BiLSTM encoder
    x = keras.layers.Bidirectional(keras.layers.LSTM(100, activation=activation, return_sequences=True))(input_series)
    encoded = keras.layers.Bidirectional(
        keras.layers.LSTM(50, activation=activation, dropout=0.2, return_sequences=False))(x)

    encoder = keras.Model(input_series, encoded)

    # BiLSTM decoder
    x = keras.layers.RepeatVector(sequence_length)(encoded)
    x = keras.layers.Bidirectional(keras.layers.LSTM(50, activation=activation, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(100, activation=activation, return_sequences=True))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dense(100, activation=activation))(x)

    decoded = keras.layers.TimeDistributed(keras.layers.Dense(num_features))(x)

    # Autoencoder
    autoencoder = keras.Model(input_series, decoded)

    return encoder, autoencoder


def define_cnn_autoencoder(sequence_length: int, num_features: int, activation: str = 'tanh'):
    # Model input
    input_series = keras.Input(shape=(sequence_length, num_features))

    # CNN encoder
    x = keras.layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation=activation)(input_series)
    x = keras.layers.Dropout(rate=0.2)(x)
    encoded = keras.layers.Conv1D(filters=16, kernel_size=7, padding="same", strides=2, activation=activation)(x)
    encoder = keras.Model(input_series, encoded)

    # CNN decoder
    x = keras.layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", strides=2, activation=activation)(
        encoded)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation=activation)(x)
    decoded = keras.layers.Conv1DTranspose(filters=num_features, kernel_size=7, padding="same")(x)

    # Autoencoder
    autoencoder = keras.Model(input_series, decoded)

    return encoder, autoencoder


def define_fcnn_autoencoder(sequence_length: int, num_features: int, activation: str = 'tanh'):
    # Model input
    input_series = keras.Input(shape=(sequence_length, num_features))

    # Fully connected neural network encoder
    x = keras.layers.Flatten()(input_series)
    x = keras.layers.Dense(200, activation=activation)(x)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Dense(200, activation=activation)(x)
    encoded = keras.layers.Dense(100, activation=activation)(x)
    encoder = keras.Model(input_series, encoded)

    # Fully connected neural network decoder
    x = keras.layers.Dense(100, activation=activation)(encoded)
    x = keras.layers.Dropout(rate=0.2)(x)
    x = keras.layers.Dense(200, activation=activation)(x)
    x = keras.layers.Dense(num_features * sequence_length)(x)
    decoded = keras.layers.Reshape((sequence_length, num_features))(x)

    # Autoencoder
    autoencoder = keras.Model(input_series, decoded)

    return encoder, autoencoder


class AutoEncoder(object):

    def __init__(self, model_type: str, activation: str = 'tanh', loss: str = 'mse', lr: float = 0.0004,
                 alpha: float = 0.02):
        """
        AutoEncoder models for novelty detection on multivariate time series

        Arguments
        ---------
        model_type : {‘cnn’, ‘fcnn’, ‘lstm’, 'bilstm'}
            Layers used to build the autoencoder architecture.

        activation : {‘tanh’, ‘relu’, ‘sigmoid ’}, default='tanh'
            Activation function for the hidden layer.
            https://keras.io/api/layers/activations/

        loss : {‘mae’, ‘mse’}, default='mse'
            Function used to compute the reconstruction error
            https://keras.io/api/losses/regression_losses/

        lr : float, default=0.0004
            The initial learning rate used. It controls the step-size in updating the weights
            https://keras.io/api/optimizers/

        alpha : float, default=0.02
            Factor to add at the trained threshold  to detect novel condition
        """

        # Multivariate time series params
        self.sequence_length = None
        self.num_features = None
        self.num_class = None

        # AutoEncoder params
        self.lr = lr  # learning rate
        self.loss = loss  # loss function
        self.activation = activation  # activation function

        # AutoEncoder and Classifier model
        self.model_type = model_type  # Base model used to build the autoencoder
        self.encoder = None
        self.autoencoder = None
        self.classifier = None

        # Training history
        self.history_autoencoder = None
        self.history_classifier = None

        # Novelty detection threshold
        self.alpha = alpha
        self.threshold = 0

        self._check_model()

    def _set_input(self, x: np.array):
        assert len(x.shape) == 3, 'Invalid input shape'
        self.sequence_length = x.shape[1]
        self.num_features = x.shape[2]

    def _set_classes(self, y: np.array):
        assert len(y.shape) == 1, 'Invalid label shape'
        self.num_class = len(np.unique(y))

    def _check_model(self, ):
        if self.model_type not in ['cnn', 'lstm', 'bilstm', 'fcnn']:
            raise ValueError("Selected the wrong model, please select one of: 'cnn', 'lstm', 'bilstm', 'fcnn'")

    def _define_autoencoder(self, ):
        if self.model_type == 'lstm':
            self.encoder, self.autoencoder = define_lstm_autoencoder(self.sequence_length, self.num_features,
                                                                     self.activation)
        elif self.model_type == 'bilstm':
            self.encoder, self.autoencoder = define_bilstm_autoencoder(self.sequence_length, self.num_features,
                                                                       self.activation)
        elif self.model_type == 'cnn':
            self.encoder, self.autoencoder = define_cnn_autoencoder(self.sequence_length, self.num_features,
                                                                    self.activation)
        elif self.model_type == 'fcnn':
            self.encoder, self.autoencoder = define_fcnn_autoencoder(self.sequence_length, self.num_features,
                                                                     self.activation)
        else:
            raise ValueError("Selected the wrong model, please select one of: 'cnn', 'lstm', 'bilstm', 'fcnn'")

        optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0, clipvalue=0.5)
        self.autoencoder.compile(optimizer=optimizer, loss=self.loss)

    def _define_classifier(self, ):
        # Classifier architecture
        input_series = keras.Input(shape=(self.sequence_length, self.num_features))

        x = self.encoder(input_series)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(200, activation='tanh')(x)
        x = keras.layers.Dense(64, activation='tanh')(x)
        x = keras.layers.Dense(self.num_class, activation='softmax')(x)

        self.classifier = keras.Model(input_series, x)
        self.encoder.trainable = False

        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        loss = keras.losses.SparseCategoricalCrossentropy()
        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    @staticmethod
    def _compute_reconstruction_error(x: np.array, x_pred: np.array):
        # Compute mean absolute error
        x = x.reshape((len(x), -1))
        x_pred = x_pred.reshape((len(x_pred), -1))
        mae_loss = np.mean(np.abs(x_pred - x), axis=1)
        return mae_loss

    def _set_reconstruction_error(self, x: np.array):
        # Reconstruct input series
        x_pred = self.autoencoder.predict(x)

        # Compute reconstruction error with mean absolute error formula
        train_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Get reconstruction error threshold.
        threshold = np.max(train_mae_loss)
        self.threshold = threshold + self.alpha

    def fit(self, x: np.array, y: np.array = None, epochs: int = 100, batch_size: int = 128, verbose: int = 0):
        """ Train autoencoder model with the optional classifier """
        # Define autoencoder input and architecture
        self._set_input(x)
        self._define_autoencoder()

        # Train the autoencoder model
        self.history_autoencoder = self.autoencoder.fit(
            x=x, y=x,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
            ],
        )

        # Save the reconstruction error
        self._set_reconstruction_error(x)

        # Classification task
        if y is not None:
            # Define classifier label and architecture
            self._set_classes(y)
            self._define_classifier()

            # Train Classifier
            self.history_classifier = self.classifier.fit(
                x=x, y=y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=verbose,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")
                ],
            )

    def predict(self, x: np.array):
        """ Detect novel conditions """
        # Input reconstruction
        x_pred = self.autoencoder.predict(x, verbose=0)

        # Reconstruction error
        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect all samples which are novel conditions
        novelties = (test_mae_loss > self.threshold).astype('int')
        return novelties

    def locate(self, x: np.array):
        """ Detect the most novel signals for each record """
        # Input reconstruction
        x_pred = self.autoencoder.predict(x)

        # Swap feature and time axes
        x = np.swapaxes(x, 1, 2)
        x_pred = np.swapaxes(x_pred, 1, 2)

        # Compute feature error
        diff = np.mean(np.abs(x - x_pred), axis=2)
        positions = diff.argmax(axis=1)

        return positions

    def decision_score(self, x: np.array):
        """ Compute and return the novel score for each record"""
        # Input reconstruction
        x_pred = self.autoencoder.predict(x)

        # Reconstruction error
        test_mae_loss = self._compute_reconstruction_error(x, x_pred)

        # Detect the samples shift from the trained threshold
        scores = test_mae_loss - self.threshold
        return scores

    def classify(self, x: np.array, supervised: bool = False, n_clusters: int = None):
        """ Classify the state of the input"""
        if supervised:
            # Supervised classifier prediction
            assert self.classifier is not None, 'Please train the supervised classifier'
            y_pred = self.classifier.predict(x, verbose=0)
            y_pred = y_pred.argmax(axis=1)
            return y_pred
        elif not supervised and n_clusters:
            # Unsupervised classifier prediction
            enc_pred = self.encoder.predict(x, verbose=0)  # Encoder output
            enc_pred = enc_pred.reshape((len(enc_pred), -1))  # Reshape encoding for the clustering step

            # Clustering with fixed number of clusters
            cls = AgglomerativeClustering(n_clusters=n_clusters)
            y_pred = cls.fit_predict(enc_pred)
            return y_pred
        else:
            assert False, 'Not supported yet'
