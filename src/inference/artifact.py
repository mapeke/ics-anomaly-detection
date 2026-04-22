"""Model-artifact bundle: model + scaler + threshold + training provenance.

An artifact is a self-contained directory that the inference pipeline (and
the web app) can point at to reproduce the exact scoring behaviour of a
past experiment run. On-disk layout::

    <dir>/
        model/                <- produced by AnomalyDetector.save
            meta.json
            hyperparams.json
            model.joblib | model.pt
        scaler.joblib         <- MinMaxScaler fitted on train-normal only
        threshold.json        <- {"value": float, "strategy": str, "percentile": float}
        manifest.json         <- provenance (see :class:`ModelArtifact`)
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler

from src.models import AnomalyDetector, load_model


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class ModelArtifact:
    """Everything needed to reproduce a trained detector's inference path."""

    model: AnomalyDetector
    scaler: MinMaxScaler
    threshold: float
    threshold_strategy: str
    feature_columns: list[str]
    trained_on: str                     # e.g. "hai" | "morris" | "hai__to__morris"
    config_hash: str
    seed: int
    window: int | None = None
    stride: int = 1
    threshold_percentile: float | None = None
    git_sha: str = field(default_factory=_git_sha)
    extra: dict = field(default_factory=dict)

    def flag(self, scores) -> "np.ndarray":  # noqa: F821 - doc only; avoid np import
        """Return ``scores >= threshold`` as a boolean array."""
        import numpy as np

        return np.asarray(scores) >= self.threshold


def save_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    """Persist an artifact directory at ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    artifact.model.save(path / "model")
    joblib.dump(artifact.scaler, path / "scaler.joblib")

    (path / "threshold.json").write_text(
        json.dumps(
            {
                "value": float(artifact.threshold),
                "strategy": artifact.threshold_strategy,
                "percentile": artifact.threshold_percentile,
            }
        )
    )

    (path / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": artifact.model.name,
                "feature_columns": list(artifact.feature_columns),
                "window": artifact.window,
                "stride": artifact.stride,
                "trained_on": artifact.trained_on,
                "config_hash": artifact.config_hash,
                "seed": artifact.seed,
                "git_sha": artifact.git_sha,
                "extra": artifact.extra,
            },
            indent=2,
        )
    )
    return path


def load_artifact(path: str | Path, device: str = "cpu") -> ModelArtifact:
    """Load an artifact directory written by :func:`save_artifact`."""
    path = Path(path)

    manifest = json.loads((path / "manifest.json").read_text())
    threshold_json = json.loads((path / "threshold.json").read_text())

    # Torch models accept ``device``; sklearn models reject it. The registry
    # loader forwards kwargs, so only pass device when the model is a deep
    # model (the model's ``meta.json`` says which).
    model_meta = json.loads((path / "model" / "meta.json").read_text())
    deep_names = {"dense_ae", "lstm_ae", "usad", "tranad"}
    load_kwargs = {"device": device} if model_meta["name"] in deep_names else {}
    model = load_model(path / "model", **load_kwargs)

    scaler = joblib.load(path / "scaler.joblib")

    return ModelArtifact(
        model=model,
        scaler=scaler,
        threshold=float(threshold_json["value"]),
        threshold_strategy=threshold_json["strategy"],
        threshold_percentile=threshold_json.get("percentile"),
        feature_columns=list(manifest["feature_columns"]),
        window=manifest.get("window"),
        stride=manifest.get("stride", 1),
        trained_on=manifest["trained_on"],
        config_hash=manifest["config_hash"],
        seed=int(manifest["seed"]),
        git_sha=manifest.get("git_sha", "unknown"),
        extra=manifest.get("extra", {}),
    )
