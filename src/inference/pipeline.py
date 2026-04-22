"""Inference pipeline: apply a :class:`ModelArtifact` to a new DataFrame.

Pure function with no side effects — the CLI (``scripts/score_external``)
and the FastAPI web app both call :func:`score_dataframe` and format the
result themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import etapr_f1, point_adjust_f1, pointwise_metrics
from src.inference.artifact import ModelArtifact
from src.preprocessing import make_windows, window_labels


@dataclass
class ScoreResult:
    """Per-sample anomaly scores and flags, optionally with labelled metrics.

    ``scores`` / ``flags`` / ``labels`` all share the same leading length:
    ``T`` for tabular models, ``N = (T - window) // stride + 1`` for windowed
    models.
    """

    scores: np.ndarray                       # (T,) or (N,)
    flags: np.ndarray                        # bool, same shape as scores
    labels: np.ndarray | None                # same shape as scores, or None
    metrics: dict[str, dict[str, Any]] | None  # pointwise/point_adjust/etapr
    n_input_rows: int                        # rows in the uploaded DataFrame
    windowed: bool


def _align_features(
    df: pd.DataFrame, expected: list[str]
) -> np.ndarray:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing expected feature columns: {missing}")
    return df[expected].to_numpy(dtype=np.float32, copy=True)


def score_dataframe(
    artifact: ModelArtifact,
    df: pd.DataFrame,
    labels: np.ndarray | None = None,
) -> ScoreResult:
    """Score ``df`` with ``artifact``'s model/scaler/threshold.

    ``df`` must contain at least the columns in ``artifact.feature_columns``
    (the adapter is responsible for ensuring this). ``labels`` is an optional
    (T,) binary array; when given, the returned ``metrics`` dict contains
    ``pointwise`` / ``point_adjust`` / ``etapr`` sub-dicts computed via the
    existing :mod:`src.evaluation` functions.
    """
    n_input = len(df)
    X_raw = _align_features(df, artifact.feature_columns)
    X = artifact.scaler.transform(X_raw).astype(np.float32)

    y = np.asarray(labels, dtype=np.int8) if labels is not None else None

    windowed = artifact.window is not None
    if windowed:
        X = make_windows(X, artifact.window, artifact.stride)
        if y is not None:
            y = window_labels(y, artifact.window, artifact.stride)

    scores = np.asarray(artifact.model.score(X))
    flags = scores >= artifact.threshold

    metrics: dict[str, dict[str, Any]] | None = None
    if y is not None:
        metrics = {
            "pointwise": pointwise_metrics(y, scores, artifact.threshold).as_dict(),
            "point_adjust": point_adjust_f1(y, scores, artifact.threshold).as_dict(),
            "etapr": etapr_f1(y, scores, artifact.threshold).as_dict(),
        }

    return ScoreResult(
        scores=scores,
        flags=flags,
        labels=y,
        metrics=metrics,
        n_input_rows=n_input,
        windowed=windowed,
    )
