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
from src.preprocessing import make_windows, percentile_threshold, window_labels

RECALIBRATE_TARGET_VAL = "target_val_percentile"
RECALIBRATE_MODES = {None, RECALIBRATE_TARGET_VAL}


@dataclass
class ScoreResult:
    """Per-sample anomaly scores and flags, optionally with labelled metrics.

    ``scores`` / ``flags`` / ``labels`` all share the same leading length:
    ``T`` for tabular models, ``N = (T - window) // stride + 1`` for windowed
    models.

    When ``recalibrate_mode`` is set, ``threshold`` reflects the recalibrated
    value and ``source_threshold`` is preserved for reference — this matches
    the ``src-cal`` vs. ``tgt-cal`` reporting convention from
    ``thesis/chapters/05_results_transfer.tex``.
    """

    scores: np.ndarray                          # (T,) or (N,)
    flags: np.ndarray                           # bool, same shape as scores
    labels: np.ndarray | None                   # same shape as scores, or None
    metrics: dict[str, dict[str, Any]] | None   # pointwise/point_adjust/etapr
    n_input_rows: int                           # rows in the uploaded DataFrame
    windowed: bool
    threshold: float                            # the threshold actually applied
    source_threshold: float                     # artifact.threshold (always preserved)
    recalibrate_mode: str | None = None
    recalibrate_percentile: float | None = None


def _align_features(
    df: pd.DataFrame, expected: list[str]
) -> np.ndarray:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing expected feature columns: {missing}")
    return df[expected].to_numpy(dtype=np.float32, copy=True)


def _pick_recalibrated_threshold(
    scores: np.ndarray,
    labels: np.ndarray | None,
    mode: str,
    percentile: float,
) -> float:
    """Return a threshold derived from the uploaded data.

    For ``target_val_percentile`` we take the ``percentile``-th percentile
    of scores on known-normal rows (``labels == 0`` when available, all
    rows otherwise). This mirrors ``experiments/run_transfer.py:171-186``.
    """
    if mode != RECALIBRATE_TARGET_VAL:
        raise ValueError(f"Unknown recalibrate mode '{mode}'; expected one of {RECALIBRATE_MODES}")
    if labels is not None:
        normal_mask = labels == 0
        if normal_mask.any():
            return percentile_threshold(scores[normal_mask], percentile=percentile)
    return percentile_threshold(scores, percentile=percentile)


def score_dataframe(
    artifact: ModelArtifact,
    df: pd.DataFrame,
    labels: np.ndarray | None = None,
    recalibrate: str | None = None,
    percentile: float = 99.0,
) -> ScoreResult:
    """Score ``df`` with ``artifact``'s model/scaler/threshold.

    ``df`` must contain at least the columns in ``artifact.feature_columns``
    (the adapter is responsible for ensuring this). ``labels`` is an optional
    (T,) binary array; when given, the returned ``metrics`` dict contains
    ``pointwise`` / ``point_adjust`` / ``etapr`` sub-dicts computed via the
    existing :mod:`src.evaluation` functions.

    ``recalibrate="target_val_percentile"`` recomputes the threshold from
    the uploaded data's normal-only scores (or all scores when no labels).
    This matches the ``tgt-cal`` protocol used in Chapter~5 of the thesis.
    """
    if recalibrate not in RECALIBRATE_MODES:
        raise ValueError(f"recalibrate must be one of {RECALIBRATE_MODES}, got '{recalibrate}'")

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

    threshold = float(artifact.threshold)
    if recalibrate is not None:
        threshold = _pick_recalibrated_threshold(scores, y, recalibrate, percentile)

    flags = scores >= threshold

    metrics: dict[str, dict[str, Any]] | None = None
    if y is not None:
        metrics = {
            "pointwise": pointwise_metrics(y, scores, threshold).as_dict(),
            "point_adjust": point_adjust_f1(y, scores, threshold).as_dict(),
            "etapr": etapr_f1(y, scores, threshold).as_dict(),
        }

    return ScoreResult(
        scores=scores,
        flags=flags,
        labels=y,
        metrics=metrics,
        n_input_rows=n_input,
        windowed=windowed,
        threshold=threshold,
        source_threshold=float(artifact.threshold),
        recalibrate_mode=recalibrate,
        recalibrate_percentile=percentile if recalibrate is not None else None,
    )
