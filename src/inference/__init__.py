"""Inference artifacts and adapters.

Pure inference path: load a fitted model + its scaler + threshold and score
new data without retraining. Used by the external-validation CLI and the
web app in ``app/``.
"""
from __future__ import annotations

from .artifact import ModelArtifact, load_artifact, save_artifact
from .pipeline import ScoreResult, score_dataframe

__all__ = [
    "ModelArtifact",
    "load_artifact",
    "save_artifact",
    "ScoreResult",
    "score_dataframe",
]
