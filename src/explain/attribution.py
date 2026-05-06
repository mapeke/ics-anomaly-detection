"""Public interface for SHAP attribution on IF and OCSVM artifacts.

The dispatcher accepts a saved-artifact directory plus a row (or batch
of rows) of *scaled* features and returns per-feature contributions
under the project's anomaly-score sign convention (higher contribution
== more anomalous).

Sign convention: both ``IsolationForestAD`` and ``OneClassSVMAD`` define
``score(x) = -model.decision_function(x)`` so positive scores flag
anomalies. The SHAP backends in :mod:`._shap_backends` mirror that by
explaining ``-decision_function(x)`` directly, so a positive
contribution means the feature pushed the row toward the anomalous side.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    """Per-feature SHAP contributions for one or more rows.

    - For a single input row, ``contributions`` has shape ``(F,)`` and
      ``score`` is a Python float.
    - For a batch of ``N`` rows, ``contributions`` has shape ``(N, F)``
      and ``score`` has shape ``(N,)``.

    ``expected_value`` is SHAP's baseline (the value at zero feature
    contribution): for ``shap_tree`` it's the mean of the explainer's
    output on the training data; for ``shap_kernel`` it's the mean over
    the background sample. Additive feature attribution holds (within
    floating-point tolerance for tree, approximation error for kernel):
    ``contributions.sum(axis=-1) + expected_value ≈ score``.
    """

    feature_names: list[str]
    contributions: np.ndarray
    score: float | np.ndarray
    method: Literal["shap_tree", "shap_kernel"]
    expected_value: float


def explain_row(
    artifact_dir: Path | str,
    row: pd.Series | pd.DataFrame | np.ndarray,
    model_type: str,
) -> AttributionResult:
    """SHAP attribution for IF and OCSVM. Other model types raise.

    Args:
        artifact_dir: directory of a saved artifact (``manifest.json`` +
            ``model/``, ``scaler.joblib``, ``threshold.json``).
        row: a single row (1D array, ``pd.Series``, or 1-row DataFrame)
            or a batch (2D array or multi-row DataFrame). Values are
            assumed already scaled by the artifact's scaler.
        model_type: ``"isolation_forest"`` or ``"ocsvm"``. Anything else
            raises ``NotImplementedError`` — AE-family models should keep
            using :mod:`src.attribution`.
    """
    X, was_single = _to_2d(row)
    if model_type == "isolation_forest":
        from ._shap_backends import shap_tree
        result = shap_tree(Path(artifact_dir), X)
    elif model_type == "ocsvm":
        from ._shap_backends import shap_kernel
        result = shap_kernel(Path(artifact_dir), X)
    else:
        raise NotImplementedError(
            f"SHAP attribution is only implemented for isolation_forest and "
            f"ocsvm. Got {model_type!r}. AE-family models use "
            f"reconstruction-error attribution; see src/attribution/."
        )
    if was_single:
        result = AttributionResult(
            feature_names=result.feature_names,
            contributions=result.contributions[0],
            score=float(np.asarray(result.score)[0]),
            method=result.method,
            expected_value=result.expected_value,
        )
    return result


def _to_2d(row: pd.Series | pd.DataFrame | np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalise ``row`` to (N, F) float32. Returns (array, was_single_row)."""
    if isinstance(row, pd.Series):
        return row.to_numpy(dtype=np.float32, copy=True).reshape(1, -1), True
    if isinstance(row, pd.DataFrame):
        arr = row.to_numpy(dtype=np.float32, copy=True)
        if arr.ndim == 1:
            return arr.reshape(1, -1), True
        return arr, len(arr) == 1
    arr = np.asarray(row, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1), True
    if arr.ndim == 2:
        return arr, len(arr) == 1
    raise ValueError(f"row must be 1D or 2D, got shape {arr.shape}")
