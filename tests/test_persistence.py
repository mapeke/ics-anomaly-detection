"""Roundtrip tests for model + artifact persistence.

Each model is fit on a tiny fixture, saved, reloaded, and re-scored. Scores
must match bit-for-bit (or within float32 tolerance for torch models where
cuDNN non-determinism cannot apply because we run on CPU).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.inference import ModelArtifact, load_artifact, save_artifact
from src.models import build, load_model


@pytest.fixture
def tabular():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 6)).astype(np.float32)
    X_val = rng.normal(size=(40, 6)).astype(np.float32)
    X_test = rng.normal(size=(60, 6)).astype(np.float32)
    return X_train, X_val, X_test


@pytest.fixture
def windowed():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 10, 4)).astype(np.float32)
    X_val = rng.normal(size=(20, 10, 4)).astype(np.float32)
    X_test = rng.normal(size=(30, 10, 4)).astype(np.float32)
    return X_train, X_val, X_test


@pytest.mark.parametrize("name", ["isolation_forest", "ocsvm"])
def test_sklearn_model_save_load_roundtrip(tabular, tmp_path, name):
    X_train, X_val, X_test = tabular
    model = build(name).fit(X_train, X_val)
    original = model.score(X_test)

    model.save(tmp_path / "mdl")
    loaded = load_model(tmp_path / "mdl")
    reloaded = loaded.score(X_test)

    np.testing.assert_array_equal(original, reloaded)
    assert loaded.name == model.name


def test_dense_ae_save_load_roundtrip(tabular, tmp_path):
    X_train, X_val, X_test = tabular
    model = build(
        "dense_ae", input_dim=X_train.shape[1], hidden=(16, 8, 16), epochs=2, batch_size=32
    ).fit(X_train, X_val)
    original = model.score(X_test)

    model.save(tmp_path / "mdl")
    loaded = load_model(tmp_path / "mdl")
    reloaded = loaded.score(X_test)

    np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-6)


def test_lstm_ae_save_load_roundtrip(windowed, tmp_path):
    X_train, X_val, X_test = windowed
    window, n_feat = X_train.shape[1:]
    model = build(
        "lstm_ae", window=window, n_features=n_feat, hidden_dim=16, latent_dim=8,
        epochs=2, batch_size=16,
    ).fit(X_train, X_val)
    original = model.score(X_test)

    model.save(tmp_path / "mdl")
    loaded = load_model(tmp_path / "mdl")
    reloaded = loaded.score(X_test)

    np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["usad", "tranad"])
def test_deep_sota_save_load_roundtrip(windowed, tmp_path, name):
    X_train, X_val, X_test = windowed
    window, n_feat = X_train.shape[1:]
    kwargs = dict(window=window, n_features=n_feat, epochs=2, batch_size=16)
    if name == "tranad":
        kwargs.update(d_model=16, n_heads=2, ff_dim=32)
    elif name == "usad":
        kwargs.update(hidden=32, latent=8)
    model = build(name, **kwargs).fit(X_train, X_val)
    original = model.score(X_test)

    model.save(tmp_path / "mdl")
    loaded = load_model(tmp_path / "mdl")
    reloaded = loaded.score(X_test)

    np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-6)


def test_artifact_save_load_roundtrip(tabular, tmp_path):
    X_train, X_val, X_test = tabular

    scaler = MinMaxScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = build("isolation_forest").fit(X_train_s, X_val_s)
    val_scores = model.score(X_val_s)
    threshold = float(np.percentile(val_scores, 99.0))

    artifact = ModelArtifact(
        model=model,
        scaler=scaler,
        threshold=threshold,
        threshold_strategy="val_percentile",
        threshold_percentile=99.0,
        feature_columns=[f"f{i}" for i in range(X_train.shape[1])],
        trained_on="synthetic",
        config_hash="abc123",
        seed=42,
    )

    save_artifact(artifact, tmp_path / "art")
    reloaded = load_artifact(tmp_path / "art")

    # Scores identical
    original_scores = model.score(X_test_s)
    reloaded_scores = reloaded.model.score(reloaded.scaler.transform(X_test).astype(np.float32))
    np.testing.assert_array_equal(original_scores, reloaded_scores)

    # Threshold / provenance survive the round trip
    assert reloaded.threshold == threshold
    assert reloaded.threshold_strategy == "val_percentile"
    assert reloaded.threshold_percentile == 99.0
    assert reloaded.feature_columns == artifact.feature_columns
    assert reloaded.trained_on == "synthetic"
    assert reloaded.config_hash == "abc123"
    assert reloaded.seed == 42

    # Flag helper matches threshold semantics
    flags = reloaded.flag(reloaded_scores)
    np.testing.assert_array_equal(flags, reloaded_scores >= threshold)
