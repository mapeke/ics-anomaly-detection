"""LSTM sequence-to-sequence autoencoder."""
from __future__ import annotations

import numpy as np


class LSTMAutoencoderAD:
    def __init__(
        self,
        window: int,
        n_features: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 128,
        verbose: int = 0,
    ):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        self.window = window
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        inputs = layers.Input(shape=(window, n_features))
        encoded = layers.LSTM(hidden_dim, return_sequences=False)(inputs)
        encoded = layers.Dense(latent_dim, activation="relu")(encoded)
        repeat = layers.RepeatVector(window)(encoded)
        decoded = layers.LSTM(hidden_dim, return_sequences=True)(repeat)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

        self.model = models.Model(inputs, outputs, name="lstm_autoencoder")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None) -> "LSTMAutoencoderAD":
        val = (X_val, X_val) if X_val is not None else None
        self.history = self.model.fit(
            X,
            X,
            validation_data=val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            shuffle=True,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-window MSE averaged over (window, features)."""
        recon = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return np.mean((X - recon) ** 2, axis=(1, 2))
