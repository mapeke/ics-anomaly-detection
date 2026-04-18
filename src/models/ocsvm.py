"""One-Class SVM wrapper conforming to the AnomalyDetector ABC."""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from .base import AnomalyDetector


class OneClassSVMAD(AnomalyDetector):
    name = "ocsvm"

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        nu: float = 0.05,
        max_samples: int = 20_000,
        random_state: int = 42,
    ):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "OneClassSVMAD":
        if len(X_train) > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_train = X_train[idx]
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)
