"""AnomalyDetector base class.

Every model in this project implements this interface so the experiment
runner, cross-dataset study, and attribution analysis can treat them
uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):
    """Unified interface for all anomaly detectors.

    Shape conventions:
        X_train / X_val / X : either (T, F) tabular or (N, W, F) windowed.
            Models that require windowed input should assert 3D.
        score return: 1D (T,) or (N,) — per-timestep or per-window anomaly score.
        attribute return: same leading shape as `score`, with a trailing
            feature axis of length F — per-(feature) contribution to the score.
    """

    name: str = "AnomalyDetector"

    @abstractmethod
    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "AnomalyDetector":
        """Fit on normal-only data. Must never see attack rows.

        Returns self for chaining.
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample anomaly score; higher = more anomalous."""
        ...

    def attribute(self, X: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the anomaly score.

        Default implementation raises; models that support attribution
        (AE family, transformers) override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement attribute()."
        )

    def supports_attribution(self) -> bool:
        """Override or check via method override to enable attribution paths."""
        return type(self).attribute is not AnomalyDetector.attribute
