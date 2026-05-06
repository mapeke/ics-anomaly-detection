"""SHAP backends for the two classical baseline detectors.

Both backends explain the project's anomaly-score sign convention
(higher == more anomalous). The wrappers around sklearn's
``decision_function`` handle the sign so callers don't need to.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.inference import load_artifact

from .attribution import AttributionResult

KERNEL_BACKGROUND_ROWS = 100
KERNEL_BACKGROUND_SEED = 42
KERNEL_NSAMPLES = 128


def shap_tree(artifact_dir: Path, X: np.ndarray) -> AttributionResult:
    """SHAP attribution for an Isolation Forest artifact.

    ``shap.TreeExplainer`` on ``IsolationForest`` explains the average
    isolation path length (positive = inlier, sklearn's convention). The
    project's ``score(x) = -decision_function(x)`` flips that, so we
    negate both the SHAP values and the expected value to land on
    "higher contribution = more anomalous". Additive feature attribution
    holds exactly to floating-point tolerance.
    """
    artifact = load_artifact(artifact_dir)
    if artifact.model.name != "isolation_forest":
        raise ValueError(
            f"shap_tree expected an isolation_forest artifact, got "
            f"{artifact.model.name!r} at {artifact_dir}"
        )

    explainer = shap.TreeExplainer(artifact.model.model)
    sv = explainer.shap_values(X, check_additivity=False)
    contributions_inlier = np.asarray(sv, dtype=np.float64)
    expected_inlier = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    # Flip into the project's anomaly-score convention (higher = more anomalous).
    contributions = -contributions_inlier
    expected_value = -expected_inlier
    score = np.asarray(artifact.model.score(X), dtype=np.float64)

    return AttributionResult(
        feature_names=list(artifact.feature_columns),
        contributions=contributions,
        score=score,
        method="shap_tree",
        expected_value=expected_value,
    )


def shap_kernel(artifact_dir: Path, X: np.ndarray) -> AttributionResult:
    """SHAP attribution for a One-Class SVM artifact.

    ``shap.KernelExplainer`` is approximate; we use a 100-row
    deterministic background sample of the training-normal split, cached
    on disk per artifact (regenerated when the cache is older than the
    artifact's main model file). The wrapped predictor returns
    ``-decision_function`` so SHAP explains the project's anomaly score
    directly; no post-hoc sign flip is needed.
    """
    artifact = load_artifact(artifact_dir)
    if artifact.model.name != "ocsvm":
        raise ValueError(
            f"shap_kernel expected an ocsvm artifact, got "
            f"{artifact.model.name!r} at {artifact_dir}"
        )

    background = _kernel_background(artifact_dir, artifact)

    def predict(arr: np.ndarray) -> np.ndarray:
        return -artifact.model.model.decision_function(np.asarray(arr, dtype=np.float64))

    explainer = shap.KernelExplainer(predict, background, seed=KERNEL_BACKGROUND_SEED)
    sv = explainer.shap_values(X, nsamples=KERNEL_NSAMPLES, silent=True)
    contributions = np.asarray(sv, dtype=np.float64)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    score = np.asarray(artifact.model.score(X), dtype=np.float64)

    return AttributionResult(
        feature_names=list(artifact.feature_columns),
        contributions=contributions,
        score=score,
        method="shap_kernel",
        expected_value=expected_value,
    )


def _kernel_background(artifact_dir: Path, artifact) -> np.ndarray:
    """Return the deterministic 100-row scaled-normal background sample.

    Cached at ``<artifact_dir>/explain_background.parquet``; regenerated
    if older than the artifact's serialised model file.
    """
    cache_path = Path(artifact_dir) / "explain_background.parquet"
    model_file = _artifact_model_file(Path(artifact_dir))
    cache_fresh = (
        cache_path.exists()
        and model_file.exists()
        and cache_path.stat().st_mtime >= model_file.stat().st_mtime
    )
    if cache_fresh:
        return pd.read_parquet(cache_path).to_numpy(dtype=np.float64)

    background = _build_background_sample(artifact)
    pd.DataFrame(background, columns=artifact.feature_columns).to_parquet(
        cache_path, index=False
    )
    return background


def _build_background_sample(artifact) -> np.ndarray:
    """Draw 100 deterministic rows from the artifact's training-normal split."""
    if artifact.trained_on != "hai":
        raise NotImplementedError(
            f"SHAP-kernel background sampling is only wired for HAI artifacts; "
            f"got trained_on={artifact.trained_on!r}. Add a loader branch to "
            f"src/explain/_shap_backends._build_background_sample to extend."
        )
    from src.data_loader import load_hai

    bundle = load_hai(seed=KERNEL_BACKGROUND_SEED)
    split_arr = np.asarray(bundle.split)
    labels_arr = np.asarray(bundle.labels)
    train_normal_mask = (split_arr == "train") & (labels_arr == 0)
    feats = bundle.features.loc[train_normal_mask, list(artifact.feature_columns)]
    if len(feats) == 0:
        raise RuntimeError("No train-normal rows found for background sample.")

    rng = np.random.default_rng(KERNEL_BACKGROUND_SEED)
    idx = rng.choice(len(feats), size=min(KERNEL_BACKGROUND_ROWS, len(feats)), replace=False)
    idx.sort()
    raw = feats.iloc[idx].to_numpy(dtype=np.float32)
    scaled = artifact.scaler.transform(raw).astype(np.float64)
    return scaled


def _artifact_model_file(artifact_dir: Path) -> Path:
    """Pick the serialised model file used to age-check the background cache."""
    for name in ("model.joblib", "model.pt"):
        candidate = artifact_dir / "model" / name
        if candidate.exists():
            return candidate
    return artifact_dir / "model" / "meta.json"
