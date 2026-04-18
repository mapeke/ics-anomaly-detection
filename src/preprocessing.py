"""Preprocessing: scaling (normal-only fit, asserted), windowing, thresholds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .data_loader import DatasetBundle


@dataclass
class ScaledArrays:
    """Scaled numeric arrays aligned to a DatasetBundle's splits."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: MinMaxScaler


def scale_bundle(bundle: DatasetBundle) -> ScaledArrays:
    """Fit MinMaxScaler on the *train* split only; transform all splits.

    Asserts that the train and val splits contain no attack rows. This is the
    single most common source of inflated numbers in published ICS AD papers
    so we guard it here rather than trusting caller discipline.
    """
    bundle.assert_no_attack_in_train_val()

    feature_names = list(bundle.features.columns)
    X_train = bundle.X("train")
    X_val = bundle.X("val")
    X_test = bundle.X("test")
    y_test = bundle.y("test")

    scaler = MinMaxScaler(clip=True)
    scaler.fit(X_train)
    return ScaledArrays(
        X_train=scaler.transform(X_train).astype(np.float32),
        X_val=scaler.transform(X_val).astype(np.float32),
        X_test=scaler.transform(X_test).astype(np.float32),
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def make_windows(X: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Convert (T, F) matrix into (N, window, F) sliding-window tensor."""
    if X.ndim != 2:
        raise ValueError(f"expected 2D input, got shape {X.shape}")
    T, F = X.shape
    if T < window:
        raise ValueError(f"series too short ({T}) for window {window}")
    n = (T - window) // stride + 1
    out = np.empty((n, window, F), dtype=X.dtype)
    for i in range(n):
        start = i * stride
        out[i] = X[start : start + window]
    return out


def window_labels(y: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """A window is labelled 1 iff any frame inside it is anomalous."""
    T = len(y)
    n = (T - window) // stride + 1
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        start = i * stride
        out[i] = int(y[start : start + window].any())
    return out


def percentile_threshold(val_scores: np.ndarray, percentile: float = 99.0) -> float:
    """Threshold = p-th percentile of validation (normal) anomaly scores.

    The "no-attack-in-validation" protocol: anything above the Pth percentile
    on clean data is considered abnormal. Avoids using test labels for
    threshold selection.
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    return float(np.percentile(val_scores, percentile))
