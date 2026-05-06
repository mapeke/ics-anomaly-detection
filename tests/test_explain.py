"""Smoke tests for SHAP attribution on IF and OCSVM artifacts.

The OCSVM background-sampling code branches on ``artifact.trained_on``
because it needs raw training data; this test monkeypatches the
background loader to a synthetic 100-row sample so we don't depend on
HAI/Morris being available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.explain import AttributionResult, explain_row
from src.inference import ModelArtifact, save_artifact
from src.models import build


def _save_synthetic_artifact(path, model_name: str, F: int = 10, T: int = 200):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, F)).astype(np.float32)
    feat_cols = [f"f{i}" for i in range(F)]

    scaler = MinMaxScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)

    if model_name == "dense_ae":
        model = build(model_name, input_dim=F, epochs=1, device="cpu").fit(Xs)
    else:
        model = build(model_name, random_state=42).fit(Xs)
    threshold = float(np.percentile(model.score(Xs), 95.0))

    artifact = ModelArtifact(
        model=model,
        scaler=scaler,
        threshold=threshold,
        threshold_strategy="val_percentile",
        threshold_percentile=95.0,
        feature_columns=feat_cols,
        trained_on="hai",
        config_hash="explain_smoke",
        seed=42,
    )
    save_artifact(artifact, path)
    sample_row = Xs[0]
    return sample_row, feat_cols


def test_shap_tree_on_isolation_forest(tmp_path):
    art_dir = tmp_path / "if_synth" / "seed42"
    row, feat_cols = _save_synthetic_artifact(art_dir, "isolation_forest")

    res = explain_row(art_dir, row, "isolation_forest")

    assert isinstance(res, AttributionResult)
    assert res.method == "shap_tree"
    assert res.feature_names == feat_cols
    assert res.contributions.shape == (len(feat_cols),)
    assert not np.isnan(res.contributions).any()
    # Note: SHAP's TreeExplainer for IsolationForest explains internal
    # path-length values, not the project's score(x) = -decision_function(x).
    # Strict additivity against `score` does not hold; what does hold is
    # that contributions rank features the same way path-length-SHAP does,
    # which is what precision@k cares about. So we don't assert additivity.


def test_shap_kernel_on_ocsvm(tmp_path, monkeypatch):
    art_dir = tmp_path / "ocsvm_synth" / "seed42"
    row, feat_cols = _save_synthetic_artifact(art_dir, "ocsvm")

    # Stub the background sampler so we don't depend on HAI being on disk.
    rng = np.random.default_rng(0)
    fake_background = rng.normal(size=(100, len(feat_cols))).astype(np.float32)
    fake_background = MinMaxScaler().fit(fake_background).transform(fake_background)

    from src.explain import _shap_backends as backends
    monkeypatch.setattr(
        backends, "_build_background_sample", lambda artifact: fake_background.astype(np.float64)
    )
    # Override KERNEL_NSAMPLES to a small value so the test is fast.
    monkeypatch.setattr(backends, "KERNEL_NSAMPLES", 64)

    res = explain_row(art_dir, row, "ocsvm")

    assert res.method == "shap_kernel"
    assert res.contributions.shape == (len(feat_cols),)
    assert not np.isnan(res.contributions).any()
    # KernelExplainer is approximate; we don't assert tight additivity.


def test_explain_row_rejects_dense_ae(tmp_path):
    """Scope guard: AE-family models must raise NotImplementedError.

    Phase 4 already covers reconstruction-error attribution for AEs; the
    SHAP backends are only for the two classical baselines that don't
    have reconstruction error.
    """
    art_dir = tmp_path / "ae_synth" / "seed42"
    row, _ = _save_synthetic_artifact(art_dir, "dense_ae")

    with pytest.raises(NotImplementedError, match="reconstruction-error"):
        explain_row(art_dir, row, "dense_ae")


def test_explain_row_accepts_batch(tmp_path):
    """A 2D row of shape (N, F) returns contributions of shape (N, F)."""
    art_dir = tmp_path / "if_batch" / "seed42"
    _, feat_cols = _save_synthetic_artifact(art_dir, "isolation_forest")
    rng = np.random.default_rng(1)
    batch = rng.uniform(size=(3, len(feat_cols))).astype(np.float32)

    res = explain_row(art_dir, batch, "isolation_forest")

    assert res.contributions.shape == (3, len(feat_cols))
    assert np.asarray(res.score).shape == (3,)
