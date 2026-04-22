"""Tests for the inference pipeline (score_dataframe + artifact roundtrip)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.inference import ModelArtifact, load_artifact, save_artifact, score_dataframe
from src.models import build


@pytest.fixture
def tabular_artifact_inputs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = np.zeros(300, dtype=np.int8)
    # Inject anomalies in known rows.
    X[50:60] += 5.0
    y[50:60] = 1

    cols = ["a", "b", "c", "d"]
    df = pd.DataFrame(X, columns=cols)

    scaler = MinMaxScaler().fit(X[:200])  # train on normal-only
    X_train_s = scaler.transform(X[:200]).astype(np.float32)
    X_val_s = scaler.transform(X[200:250]).astype(np.float32)

    model = build("isolation_forest").fit(X_train_s, X_val_s)
    val_scores = model.score(X_val_s)
    threshold = float(np.percentile(val_scores, 95.0))
    return df, y, cols, model, scaler, threshold


def test_score_dataframe_tabular(tabular_artifact_inputs):
    df, y, cols, model, scaler, threshold = tabular_artifact_inputs
    artifact = ModelArtifact(
        model=model,
        scaler=scaler,
        threshold=threshold,
        threshold_strategy="val_percentile",
        threshold_percentile=95.0,
        feature_columns=cols,
        trained_on="synthetic",
        config_hash="test",
        seed=42,
    )
    result = score_dataframe(artifact, df, labels=y)

    assert result.scores.shape == (len(df),)
    assert result.flags.dtype == bool
    assert result.metrics is not None
    for family in ("pointwise", "point_adjust", "etapr"):
        assert family in result.metrics
    assert 0.0 <= result.metrics["pointwise"]["f1"] <= 1.0


def test_score_dataframe_without_labels(tabular_artifact_inputs):
    df, _, cols, model, scaler, threshold = tabular_artifact_inputs
    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=cols, trained_on="synthetic", config_hash="x", seed=0,
    )
    result = score_dataframe(artifact, df)  # no labels

    assert result.metrics is None
    assert result.labels is None
    assert result.scores.shape == (len(df),)


def test_score_dataframe_windowed(tmp_path):
    rng = np.random.default_rng(0)
    T, F, W = 400, 3, 10
    X = rng.normal(size=(T, F)).astype(np.float32)
    y = np.zeros(T, dtype=np.int8)
    y[200:220] = 1
    X[200:220] += 4.0

    cols = [f"f{i}" for i in range(F)]
    df = pd.DataFrame(X, columns=cols)

    scaler = MinMaxScaler().fit(X[:300])
    X_train_s = scaler.transform(X[:280]).astype(np.float32)
    from src.preprocessing import make_windows, window_labels

    X_train_w = make_windows(X_train_s, W, 1)
    model = build(
        "lstm_ae", window=W, n_features=F, hidden_dim=16, latent_dim=8,
        epochs=2, batch_size=16,
    ).fit(X_train_w)

    val_scores = model.score(make_windows(scaler.transform(X[280:300]).astype(np.float32), W, 1))
    threshold = float(np.percentile(val_scores, 99.0))

    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=99.0,
        feature_columns=cols, window=W, stride=1, trained_on="synthetic",
        config_hash="win", seed=0,
    )

    out_dir = tmp_path / "art"
    save_artifact(artifact, out_dir)
    reloaded = load_artifact(out_dir)
    result = score_dataframe(reloaded, df, labels=y)

    expected_n = (T - W) // 1 + 1
    assert result.scores.shape == (expected_n,)
    assert result.labels.shape == (expected_n,)
    assert result.metrics is not None
    assert result.windowed is True


def test_score_dataframe_missing_columns_raises(tabular_artifact_inputs):
    df, _, cols, model, scaler, threshold = tabular_artifact_inputs
    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=cols, trained_on="synthetic", config_hash="x", seed=0,
    )
    truncated = df.drop(columns=["d"])
    with pytest.raises(KeyError):
        score_dataframe(artifact, truncated)


def test_artifact_roundtrip_identical_scores(tabular_artifact_inputs, tmp_path):
    df, y, cols, model, scaler, threshold = tabular_artifact_inputs
    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=cols, trained_on="synthetic", config_hash="x", seed=0,
    )
    out_dir = tmp_path / "art"
    save_artifact(artifact, out_dir)
    reloaded = load_artifact(out_dir)

    original = score_dataframe(artifact, df, labels=y)
    reloaded_result = score_dataframe(reloaded, df, labels=y)

    np.testing.assert_array_equal(original.scores, reloaded_result.scores)
    np.testing.assert_array_equal(original.flags, reloaded_result.flags)
    # Metrics may contain NaN (pr_auc when only one class present) — use
    # NaN-aware deep comparison.
    np.testing.assert_equal(original.metrics, reloaded_result.metrics)
