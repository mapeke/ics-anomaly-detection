"""Cross-dataset transfer driver.

Usage:
    python -m experiments.run_transfer experiments/configs/transfer_hai_to_morris_dense_ae.yaml

Differences vs. ``experiments.run``:
    * ``data:`` block contains ``source_dataset`` and ``target_dataset``
      (each "hai" or "morris").
    * Both datasets are loaded, projected to the type intersection
      (``data/feature_types.yaml``), then independently scaled on their
      own normal-only train splits.
    * The model fits on the *source* train/val splits and is scored on
      the *target* test split.
    * Result rows carry both ``dataset_train`` and ``dataset_test``
      columns so they sit alongside same-dataset baselines in
      ``summary.parquet``.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run import _DEEP_MODELS, _WINDOWED_MODELS, append_summary  # noqa: E402
from src.config import EvalConfig, ModelConfig, TrainConfig  # noqa: E402
from src.data_loader import DatasetBundle, load_hai, load_morris  # noqa: E402
from src.evaluation import etapr_f1, point_adjust_f1, pointwise_metrics  # noqa: E402
from src.evaluation.pointwise import best_f1_threshold  # noqa: E402
from src.models import build  # noqa: E402
from src.preprocessing import (  # noqa: E402
    make_windows,
    percentile_threshold,
    scale_bundle,
    window_labels,
)
from src.transfer import common_types, load_feature_types, project_bundle_to_types
from src.utils import set_seed  # noqa: E402

LOADERS = {"hai": load_hai, "morris": load_morris}


@dataclass
class TransferDataConfig:
    source_dataset: str
    target_dataset: str
    window: int | None = None
    stride: int = 1
    val_frac: float = 0.15
    seed: int = 42
    subsample_train: int | None = None


@dataclass
class TransferRunConfig:
    name: str
    data: TransferDataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seeds: list[int] = field(default_factory=lambda: [42])
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransferRunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            name=raw["name"],
            data=TransferDataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            train=TrainConfig(**raw.get("train", {})),
            eval=EvalConfig(**raw.get("eval", {})),
            seeds=raw.get("seeds", [42]),
            notes=raw.get("notes", ""),
        )

    def hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {
                "name": self.name,
                "data": vars(self.data),
                "model": {"name": self.model.name, "params": self.model.params},
                "train": vars(self.train),
                "eval": vars(self.eval),
                "seeds": self.seeds,
            },
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def _project_and_scale(bundle: DatasetBundle, types_yaml: dict, target_types: list[str]):
    projected = project_bundle_to_types(
        bundle, types_yaml, target_types=target_types, dataset_key=bundle.name
    )
    scaled = scale_bundle(projected)
    return scaled


def _build_model(run: TransferRunConfig, X_train: np.ndarray, seed: int):
    params = dict(run.model.params)
    if run.model.name == "dense_ae":
        params.setdefault("input_dim", X_train.shape[-1])
    if run.model.name in _WINDOWED_MODELS:
        if X_train.ndim != 3:
            raise ValueError(f"{run.model.name} requires a windowed dataset.")
        params.setdefault("window", X_train.shape[1])
        params.setdefault("n_features", X_train.shape[2])
    if run.model.name in _DEEP_MODELS:
        params.setdefault("device", run.train.device)
        params.setdefault("epochs", run.train.epochs)
        params.setdefault("batch_size", run.train.batch_size)
        params.setdefault("learning_rate", run.train.learning_rate)
    if run.model.name in {"isolation_forest", "ocsvm"}:
        params["random_state"] = seed
    return build(run.model.name, **params)


def _maybe_window(X: np.ndarray, y: np.ndarray | None, window: int | None, stride: int):
    if window is None:
        return X, y
    Xw = make_windows(X, window, stride)
    yw = window_labels(y, window, stride) if y is not None else None
    return Xw, yw


def run_once(run: TransferRunConfig, seed: int) -> list[dict]:
    set_seed(seed)
    types_yaml = load_feature_types()
    target_types = common_types(run.data.source_dataset, run.data.target_dataset) \
        if isinstance(types_yaml, list) else common_types(types_yaml, run.data.source_dataset, run.data.target_dataset)

    src_bundle = LOADERS[run.data.source_dataset](val_frac=run.data.val_frac, seed=run.data.seed)
    tgt_bundle = LOADERS[run.data.target_dataset](val_frac=run.data.val_frac, seed=run.data.seed)

    src = _project_and_scale(src_bundle, types_yaml, target_types)
    tgt = _project_and_scale(tgt_bundle, types_yaml, target_types)

    X_tr, _ = _maybe_window(src.X_train, None, run.data.window, run.data.stride)
    X_val, _ = _maybe_window(src.X_val, None, run.data.window, run.data.stride)
    X_tgt_val, _ = _maybe_window(tgt.X_val, None, run.data.window, run.data.stride)
    X_te, y_te = _maybe_window(tgt.X_test, tgt.y_test, run.data.window, run.data.stride)

    if run.data.subsample_train and len(X_tr) > run.data.subsample_train:
        rng = np.random.default_rng(seed)
        X_tr = X_tr[rng.choice(len(X_tr), run.data.subsample_train, replace=False)]

    model = _build_model(run, X_tr, seed)

    t0 = time.time()
    model.fit(X_tr, X_val)
    fit_seconds = time.time() - t0

    val_scores = model.score(X_val)
    test_scores = model.score(X_te)

    if run.eval.threshold_method == "val_percentile":
        threshold = percentile_threshold(val_scores, run.eval.val_percentile)
        threshold_source = "source_val"
    elif run.eval.threshold_method == "target_val_percentile":
        # Unlabeled target-normal calibration: score the target's val split
        # (which is attack-free by construction) and take the same percentile.
        # Isolates the 'does the representation transfer' question from the
        # 'does the source-derived operating point transfer' question.
        tgt_val_scores = model.score(X_tgt_val)
        threshold = percentile_threshold(tgt_val_scores, run.eval.val_percentile)
        threshold_source = "target_val"
    elif run.eval.threshold_method == "best_f1_oracle":
        threshold, _ = best_f1_threshold(test_scores, y_te)
        threshold_source = "test_oracle"
    else:
        raise ValueError(run.eval.threshold_method)

    base = {
        "run_name": run.name,
        "config_hash": run.hash(),
        "dataset": f"{run.data.source_dataset}__to__{run.data.target_dataset}",
        "dataset_train": run.data.source_dataset,
        "dataset_test": run.data.target_dataset,
        "model": run.model.name,
        "seed": seed,
        "threshold": float(threshold),
        "threshold_method": run.eval.threshold_method,
        "threshold_source": threshold_source,
        "fit_seconds": fit_seconds,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_attack_rate": float(y_te.mean()),
    }

    rows = []
    if "pointwise" in run.eval.metrics:
        m = pointwise_metrics(y_te, test_scores, threshold)
        rows.append({**base, "metric": "pointwise", **m.as_dict()})
    if "point_adjust" in run.eval.metrics:
        m = point_adjust_f1(y_te, test_scores, threshold)
        rows.append({**base, "metric": "point_adjust", **m.as_dict()})
    if "etapr" in run.eval.metrics:
        m = etapr_f1(y_te, test_scores, threshold)
        rows.append({**base, "metric": "etapr", **m.as_dict()})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run = TransferRunConfig.from_yaml(args.config)
    print(f">> {run.name}  (hash={run.hash()})")
    print(f"   {run.data.source_dataset} -> {run.data.target_dataset}  "
          f"model={run.model.name}  seeds={run.seeds}")
    if args.dry_run:
        return

    all_rows = []
    for seed in run.seeds:
        print(f"   seed {seed} ...")
        rows = run_once(run, seed)
        for r in rows:
            print(f"     {r['metric']:>14s}  F1={r.get('f1', r.get('etapr_f1', 0)):.4f}")
        all_rows.extend(rows)
    out = append_summary(all_rows)
    print(f"   wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
