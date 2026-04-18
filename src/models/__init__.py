"""Anomaly detection models.

All models implement :class:`~src.models.base.AnomalyDetector`.
The :data:`REGISTRY` dict is used by ``experiments/run.py`` to resolve
model names from YAML configs.
"""
from __future__ import annotations

from .autoencoder import DenseAutoencoderAD
from .base import AnomalyDetector
from .isolation_forest import IsolationForestAD
from .lstm_autoencoder import LSTMAutoencoderAD
from .ocsvm import OneClassSVMAD

REGISTRY: dict[str, type[AnomalyDetector]] = {
    IsolationForestAD.name: IsolationForestAD,
    OneClassSVMAD.name: OneClassSVMAD,
    DenseAutoencoderAD.name: DenseAutoencoderAD,
    LSTMAutoencoderAD.name: LSTMAutoencoderAD,
}


def build(name: str, **kwargs) -> AnomalyDetector:
    """Construct an anomaly detector from its registry name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "AnomalyDetector",
    "DenseAutoencoderAD",
    "IsolationForestAD",
    "LSTMAutoencoderAD",
    "OneClassSVMAD",
    "REGISTRY",
    "build",
]
