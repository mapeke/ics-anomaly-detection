"""Isolation Forest wrapper conforming to the AnomalyDetector ABC."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
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
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
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

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        (path / "hyperparams.json").write_text(
            json.dumps(
                {
                    "n_estimators": self.n_estimators,
                    "contamination": self.contamination,
                    "max_samples": self.max_samples,
                    "random_state": self.random_state,
                }
            )
        )
        (path / "meta.json").write_text(json.dumps({"name": self.name}))

    @classmethod
    def load(cls, path: Path) -> "IsolationForestAD":
        path = Path(path)
        kwargs = json.loads((path / "hyperparams.json").read_text())
        obj = cls(**kwargs)
        obj.model = joblib.load(path / "model.joblib")
        return obj
