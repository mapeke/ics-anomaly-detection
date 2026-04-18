"""One-Class SVM wrapper."""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMAD:
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

    def fit(self, X: np.ndarray) -> "OneClassSVMAD":
        if len(X) > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), self.max_samples, replace=False)
            X = X[idx]
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous."""
        return -self.model.decision_function(X)
