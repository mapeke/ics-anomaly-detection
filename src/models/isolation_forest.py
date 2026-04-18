"""Isolation Forest wrapper conforming to the AnomalyDetector ABC."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import AnomalyDetector


class IsolationForestAD(AnomalyDetector):
    name = "isolation_forest"

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

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "IsolationForestAD":
        self.model.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)
