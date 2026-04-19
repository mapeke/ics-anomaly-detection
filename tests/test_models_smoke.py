"""Smoke tests — every model fits a tiny input and produces scores/attribution."""
from __future__ import annotations

import numpy as np
import pytest

from src.models import REGISTRY, build


@pytest.fixture
def tabular():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(200, 8)).astype(np.float32)
    X_val = rng.normal(size=(30, 8)).astype(np.float32)
    X_test = rng.normal(size=(50, 8)).astype(np.float32)
    return X_train, X_val, X_test


@pytest.fixture
def windowed():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(100, 12, 5)).astype(np.float32)
    X_val = rng.normal(size=(20, 12, 5)).astype(np.float32)
    X_test = rng.normal(size=(30, 12, 5)).astype(np.float32)
    return X_train, X_val, X_test


@pytest.mark.parametrize("name", ["isolation_forest", "ocsvm"])
def test_sklearn_models_fit_score(tabular, name):
    X_train, X_val, X_test = tabular
    model = build(name)
    model.fit(X_train, X_val)
    scores = model.score(X_test)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()


def test_dense_ae_fit_score_attribute(tabular):
    X_train, X_val, X_test = tabular
    model = build("dense_ae", input_dim=X_train.shape[1], epochs=2, batch_size=32)
    model.fit(X_train, X_val)
    scores = model.score(X_test)
    attr = model.attribute(X_test)
    assert scores.shape == (len(X_test),)
    assert attr.shape == X_test.shape
    assert (attr >= 0).all()


def test_lstm_ae_fit_score_attribute(windowed):
    X_train, X_val, X_test = windowed
    window, n_feat = X_train.shape[1:]
    model = build("lstm_ae", window=window, n_features=n_feat, epochs=2, batch_size=16)
    model.fit(X_train, X_val)
    scores = model.score(X_test)
    attr = model.attribute(X_test)
    assert scores.shape == (len(X_test),)
    assert attr.shape == (len(X_test), n_feat)
    assert (attr >= 0).all()


@pytest.mark.parametrize("name", ["usad", "tranad"])
def test_sota_models_fit_score_attribute(windowed, name):
    X_train, X_val, X_test = windowed
    window, n_feat = X_train.shape[1:]
    kwargs = dict(window=window, n_features=n_feat, epochs=2, batch_size=16)
    if name == "tranad":
        kwargs.update(d_model=16, n_heads=2, ff_dim=32)
    elif name == "usad":
        kwargs.update(hidden=32, latent=8)
    model = build(name, **kwargs)
    model.fit(X_train, X_val)
    scores = model.score(X_test)
    attr = model.attribute(X_test)
    assert scores.shape == (len(X_test),)
    assert attr.shape == (len(X_test), n_feat)
    assert (attr >= 0).all()


def test_registry_is_complete():
    assert set(REGISTRY) == {
        "isolation_forest", "ocsvm", "dense_ae", "lstm_ae", "usad", "tranad",
    }


def test_base_ae_supports_attribution(tabular):
    X_train, _, _ = tabular
    ae = build("dense_ae", input_dim=X_train.shape[1], epochs=1, batch_size=32).fit(X_train)
    assert ae.supports_attribution()
    if_ = build("isolation_forest").fit(X_train)
    assert not if_.supports_attribution()
