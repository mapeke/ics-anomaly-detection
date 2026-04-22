"""Experiment configuration — YAML in, dataclasses out, config-hash for traceability."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str                           # "hai" | "morris"
    window: int | None = None              # None = tabular; int = sliding window
    stride: int = 1
    val_frac: float = 0.15
    seed: int = 42
    subsample_train: int | None = None     # cap rows for classical models


@dataclass
class ModelConfig:
    name: str                              # registry key, e.g. "lstm_ae"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    device: str = "cpu"                    # "cpu" | "cuda"


@dataclass
class EvalConfig:
    threshold_method: str = "val_percentile"   # "val_percentile" | "best_f1_oracle"
    val_percentile: float = 99.0
    metrics: list[str] = field(default_factory=lambda: ["pointwise", "point_adjust", "etapr"])


@dataclass
class ArtifactConfig:
    """Optional: persist the fitted model + scaler + threshold for reuse.

    When ``save_dir`` is set, each seed's artifact lands at
    ``<save_dir>/seed<seed>/`` so cross-seed comparisons remain possible.
    Default is off (None) — no behaviour change for existing configs.
    """

    save_dir: str | None = None


@dataclass
class RunConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    seeds: list[int] = field(default_factory=lambda: [42])
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            name=raw["name"],
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw.get("train", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            artifact=ArtifactConfig(**raw.get("artifact", {})),
            seeds=raw.get("seeds", [42]),
            notes=raw.get("notes", ""),
        )

    def hash(self) -> str:
        """Short deterministic hash over all config fields — identifies a run."""
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:12]
