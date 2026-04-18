"""Isolation Forest wrapper — anomaly score = -decision_function."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestAD:
    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        max_samples: str | int = "auto",
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "IsolationForestAD":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (flip sklearn's decision_function sign)."""
        return -self.model.decision_function(X)
