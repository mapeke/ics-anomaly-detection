"""Preprocessing invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetBundle
from src.preprocessing import make_windows, percentile_threshold, scale_bundle, window_labels


def _synth_bundle(leak: bool = False) -> DatasetBundle:
    n_train, n_val, n_test = 100, 20, 50
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(n_train + n_val + n_test, 3)).astype(np.float32)
    labels = np.zeros(len(feats), dtype=np.int8)
    # Attacks only in test — unless we deliberately leak.
    labels[-10:] = 1
    if leak:
        labels[0] = 1

    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    attack_ids = np.where(labels > 0, "synth_attack", "normal")
    return DatasetBundle(
        features=pd.DataFrame(feats, columns=["a", "b", "c"]),
        labels=labels,
        attack_ids=attack_ids,
        split=split,
        name="synth",
    )


def test_scale_bundle_fits_on_train_only():
    b = _synth_bundle()
    scaled = scale_bundle(b)
    # MinMaxScaler clip=True means train values in [0, 1].
    assert scaled.X_train.min() >= 0 - 1e-6
    assert scaled.X_train.max() <= 1 + 1e-6


def test_scale_bundle_rejects_leaked_attack_in_train():
    b = _synth_bundle(leak=True)
    with pytest.raises(AssertionError):
        scale_bundle(b)


def test_make_windows_shape():
    X = np.arange(100, dtype=np.float32).reshape(50, 2)
    W = make_windows(X, window=10, stride=5)
    assert W.shape == (9, 10, 2)
    np.testing.assert_array_equal(W[0], X[0:10])
    np.testing.assert_array_equal(W[1], X[5:15])


def test_window_labels_propagate_any_attack():
    y = np.array([0, 0, 1, 0, 0, 0])
    wl = window_labels(y, window=3, stride=1)
    # windows: [0,0,1]=1, [0,1,0]=1, [1,0,0]=1, [0,0,0]=0
    np.testing.assert_array_equal(wl, np.array([1, 1, 1, 0]))


def test_percentile_threshold_monotone_in_percentile():
    val = np.arange(100.0)
    t90 = percentile_threshold(val, percentile=90)
    t99 = percentile_threshold(val, percentile=99)
    assert t99 > t90


def test_percentile_threshold_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile_threshold(np.arange(10.0), percentile=100)
    with pytest.raises(ValueError):
        percentile_threshold(np.arange(10.0), percentile=0)
