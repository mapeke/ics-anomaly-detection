"""Leave-one-process-out (LOPO) runner for HAI.

Usage:
    python -m experiments.run_lopo experiments/configs/lopo_hai_P2_lstm_ae.yaml

Drops all ``{held_out}_*`` feature columns before fit/score so the
detector never sees the dropped process. Writes one summary row per
(metric, held_out) and reports **per-attack-target-process F1** in
separate columns so the notebook can plot the held_out x attack_process
heatmap that isolates feature-distribution shift from class-prior
shift (contrast with run_transfer.py, which hits both).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run import _DEEP_MODELS, _WINDOWED_MODELS, append_summary  # noqa: E402
from src.config import EvalConfig, ModelConfig, TrainConfig  # noqa: E402
from src.data_loader import load_hai  # noqa: E402
from src.evaluation import etapr_f1, point_adjust_f1, pointwise_metrics  # noqa: E402
from src.evaluation.pointwise import best_f1_threshold  # noqa: E402
from src.models import build  # noqa: E402
from src.preprocessing import (  # noqa: E402
    make_windows,
    percentile_threshold,
    scale_bundle,
    window_labels,
)
from src.transfer import (  # noqa: E402
    HAI_PROCESSES,
    drop_process_features,
    per_attack_process_f1,
)
from src.utils import set_seed  # noqa: E402


@dataclass
class LopoDataConfig:
    held_out: str              # "P1" | "P2" | "P3" | "P4"
    window: int | None = None
    stride: int = 1
    val_frac: float = 0.15
    seed: int = 42
    subsample_train: int | None = None


@dataclass
class LopoRunConfig:
    name: str
    data: LopoDataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seeds: list[int] = field(default_factory=lambda: [42])
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LopoRunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            name=raw["name"],
            data=LopoDataConfig(**raw["data"]),
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


def _build_model(run: LopoRunConfig, X_train: np.ndarray, seed: int):
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


def _maybe_window(X, y, window, stride):
    if window is None:
        return X, y
    Xw = make_windows(X, window, stride)
    yw = window_labels(y, window, stride) if y is not None else None
    return Xw, yw


def _window_attack_ids(attack_ids: np.ndarray, window: int | None, stride: int) -> np.ndarray:
    """Assign each window the attack_id of its *last* timestep (same convention
    as window_labels). Unwindowed: identity."""
    if window is None:
        return attack_ids
    n = len(attack_ids)
    ends = np.arange(window - 1, n, stride)
    return attack_ids[ends]


def run_once(run: LopoRunConfig, seed: int) -> list[dict]:
    set_seed(seed)
    bundle = load_hai(val_frac=run.data.val_frac, seed=run.data.seed)
    attack_ids = bundle.attack_ids
    dropped = drop_process_features(bundle, run.data.held_out)
    scaled = scale_bundle(dropped)

    test_mask = bundle.split == "test"
    test_attack_ids = attack_ids[test_mask]

    X_tr, _ = _maybe_window(scaled.X_train, None, run.data.window, run.data.stride)
    X_val, _ = _maybe_window(scaled.X_val, None, run.data.window, run.data.stride)
    X_te, y_te = _maybe_window(
        scaled.X_test, scaled.y_test, run.data.window, run.data.stride,
    )
    windowed_test_aids = _window_attack_ids(test_attack_ids, run.data.window, run.data.stride)

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
    elif run.eval.threshold_method == "best_f1_oracle":
        threshold, _ = best_f1_threshold(test_scores, y_te)
    else:
        raise ValueError(run.eval.threshold_method)

    y_pred = (test_scores >= threshold).astype(np.int8)
    per_proc = per_attack_process_f1(y_te, y_pred, windowed_test_aids)

    base = {
        "run_name": run.name,
        "config_hash": run.hash(),
        "dataset": "hai__lopo",
        "dataset_train": "hai",
        "dataset_test": "hai",
        "held_out_process": run.data.held_out,
        "model": run.model.name,
        "seed": seed,
        "threshold": float(threshold),
        "threshold_method": run.eval.threshold_method,
        "fit_seconds": fit_seconds,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_attack_rate": float(y_te.mean()),
        "n_features": int(scaled.X_train.shape[-1]),
        **{f"f1_attack_{p}": per_proc[p] for p in HAI_PROCESSES},
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

    run = LopoRunConfig.from_yaml(args.config)
    print(f">> {run.name}  (hash={run.hash()})")
    print(f"   hai \\ {run.data.held_out}  model={run.model.name}  seeds={run.seeds}")
    if args.dry_run:
        return

    all_rows = []
    for seed in run.seeds:
        print(f"   seed {seed} ...")
        rows = run_once(run, seed)
        for r in rows:
            per = " ".join(f"{p}={r[f'f1_attack_{p}']:.2f}" for p in HAI_PROCESSES)
            f1_main = r.get('f1', r.get('etapr_f1', 0))
            print(f"     {r['metric']:>14s}  F1={f1_main:.4f}  |  per-proc: {per}")
        all_rows.extend(rows)
    out = append_summary(all_rows)
    print(f"   wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
