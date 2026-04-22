"""Anomaly detection models.

All models implement :class:`~src.models.base.AnomalyDetector`.
The :data:`REGISTRY` dict is used by ``experiments/run.py`` to resolve
model names from YAML configs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .autoencoder import DenseAutoencoderAD
from .base import AnomalyDetector
from .isolation_forest import IsolationForestAD
from .lstm_autoencoder import LSTMAutoencoderAD
from .ocsvm import OneClassSVMAD
from .tranad import TranADModel
from .usad import USADModel

REGISTRY: dict[str, type[AnomalyDetector]] = {
    IsolationForestAD.name: IsolationForestAD,
    OneClassSVMAD.name: OneClassSVMAD,
    DenseAutoencoderAD.name: DenseAutoencoderAD,
    LSTMAutoencoderAD.name: LSTMAutoencoderAD,
    USADModel.name: USADModel,
    TranADModel.name: TranADModel,
}


def build(name: str, **kwargs) -> AnomalyDetector:
    """Construct an anomaly detector from its registry name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def load_model(path: Path, **load_kwargs) -> AnomalyDetector:
    """Load a fitted model from ``path`` without needing its concrete class.

    Reads ``meta.json`` to pick the right :data:`REGISTRY` entry, then
    delegates to that class's :meth:`AnomalyDetector.load`. ``load_kwargs``
    (e.g. ``device="cuda"``) are passed through.
    """
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    name = meta["name"]
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}' in {path / 'meta.json'}.")
    return REGISTRY[name].load(path, **load_kwargs)


__all__ = [
    "AnomalyDetector",
    "DenseAutoencoderAD",
    "IsolationForestAD",
    "LSTMAutoencoderAD",
    "OneClassSVMAD",
    "TranADModel",
    "USADModel",
    "REGISTRY",
    "build",
    "load_model",
]
