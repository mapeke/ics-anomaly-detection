"""Dense autoencoder — reconstruction MSE as anomaly score."""
from __future__ import annotations

import numpy as np


class DenseAutoencoderAD:
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (64, 32, 16, 32, 64),
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        epochs: int = 40,
        batch_size: int = 256,
        verbose: int = 0,
    ):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for units in hidden:
            x = layers.Dense(units, activation="relu")(x)
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
        outputs = layers.Dense(input_dim, activation="sigmoid")(x)

        self.model = models.Model(inputs, outputs, name="dense_autoencoder")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )

    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None) -> "DenseAutoencoderAD":
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
        recon = self.model.predict(X, batch_size=self.batch_size, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)
