"""Preprocessing: scaling, sliding windows, train/val splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ScaledData:
    """Container for scaled training/validation/test arrays and their labels."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: MinMaxScaler


def prepare_tabular(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "label",
    val_frac: float = 0.15,
    seed: int = 42,
) -> ScaledData:
    """Scale features to [0, 1] using statistics from the normal training set.

    Assumes train_df contains only normal data (label == 0).
    """
    features = [c for c in train_df.columns if c != label_col]
    # Keep only columns present in both, in a stable order.
    features = [c for c in features if c in test_df.columns]

    X_train_full = train_df[features].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    y_test = test_df[label_col].to_numpy(dtype=np.int32)

    rng = np.random.default_rng(seed)
    n_val = int(len(X_train_full) * val_frac)
    idx = rng.permutation(len(X_train_full))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    scaler = MinMaxScaler(clip=True)
    X_train = scaler.fit_transform(X_train_full[train_idx])
    X_val = scaler.transform(X_train_full[val_idx])
    X_test = scaler.transform(X_test)

    return ScaledData(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_test=y_test,
        feature_names=features,
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
