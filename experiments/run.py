"""Single-experiment driver.

Usage:
    python -m experiments.run experiments/configs/baseline_hai_lstm_ae.yaml

Reads a RunConfig from YAML, executes the model on the configured dataset
for each seed, applies the configured threshold protocol, and writes one
row per (run, seed, metric) into ``results/metrics/summary.parquet``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RunConfig
from src.data_loader import DatasetBundle, load_hai, load_morris
from src.evaluation import etapr_f1, point_adjust_f1, pointwise_metrics
from src.evaluation.pointwise import best_f1_threshold
from src.models import build
from src.preprocessing import make_windows, percentile_threshold, scale_bundle, window_labels
from src.utils import METRICS_DIR, set_seed

SUMMARY_PATH = METRICS_DIR / "summary.parquet"


def _load_dataset(cfg) -> DatasetBundle:
    if cfg.dataset == "hai":
        return load_hai(val_frac=cfg.val_frac, seed=cfg.seed)
    if cfg.dataset == "morris":
        return load_morris(val_frac=cfg.val_frac, seed=cfg.seed)
    raise ValueError(f"unknown dataset '{cfg.dataset}'")


def _prepare_arrays(run: RunConfig):
    """Return (X_train, X_val, X_test, y_test) possibly windowed, plus ctx."""
    bundle = _load_dataset(run.data)
    scaled = scale_bundle(bundle)

    X_train, X_val, X_test, y_test = scaled.X_train, scaled.X_val, scaled.X_test, scaled.y_test

    if run.data.subsample_train and len(X_train) > run.data.subsample_train:
        rng = np.random.default_rng(run.data.seed)
        X_train = X_train[rng.choice(len(X_train), run.data.subsample_train, replace=False)]

    if run.data.window is not None:
        X_train = make_windows(X_train, run.data.window, run.data.stride)
        X_val = make_windows(X_val, run.data.window, run.data.stride)
        X_test_w = make_windows(X_test, run.data.window, run.data.stride)
        y_test = window_labels(y_test, run.data.window, run.data.stride)
        X_test = X_test_w

    return X_train, X_val, X_test, y_test, scaled.feature_names


def _build_model(run: RunConfig, X_train: np.ndarray, seed: int):
    params = dict(run.model.params)
    # Auto-fill input-dim / window / n_features if the model needs them.
    if run.model.name == "dense_ae":
        params.setdefault("input_dim", X_train.shape[-1])
        params.setdefault("device", run.train.device)
    elif run.model.name == "lstm_ae":
        if X_train.ndim != 3:
            raise ValueError("lstm_ae requires a windowed dataset (data.window set).")
        params.setdefault("window", X_train.shape[1])
        params.setdefault("n_features", X_train.shape[2])
        params.setdefault("device", run.train.device)
    if run.model.name in {"dense_ae", "lstm_ae"}:
        params.setdefault("epochs", run.train.epochs)
        params.setdefault("batch_size", run.train.batch_size)
        params.setdefault("learning_rate", run.train.learning_rate)
    # Propagate the experiment seed so classical models vary across runs.
    if run.model.name in {"isolation_forest", "ocsvm"}:
        params["random_state"] = seed
    return build(run.model.name, **params)


def _pick_threshold(method: str, val_scores: np.ndarray, test_scores: np.ndarray,
                    y_test: np.ndarray, percentile: float) -> float:
    if method == "val_percentile":
        return percentile_threshold(val_scores, percentile=percentile)
    if method == "best_f1_oracle":
        thr, _ = best_f1_threshold(test_scores, y_test)
        return thr
    raise ValueError(f"unknown threshold method '{method}'")


def run_once(run: RunConfig, seed: int) -> list[dict]:
    set_seed(seed)
    X_train, X_val, X_test, y_test, _ = _prepare_arrays(run)
    model = _build_model(run, X_train, seed)

    t0 = time.time()
    model.fit(X_train, X_val)
    fit_seconds = time.time() - t0

    val_scores = model.score(X_val)
    test_scores = model.score(X_test)
    threshold = _pick_threshold(
        run.eval.threshold_method, val_scores, test_scores, y_test, run.eval.val_percentile
    )

    rows: list[dict] = []
    base = {
        "run_name": run.name,
        "config_hash": run.hash(),
        "dataset": run.data.dataset,
        "model": run.model.name,
        "seed": seed,
        "threshold": float(threshold),
        "threshold_method": run.eval.threshold_method,
        "fit_seconds": fit_seconds,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_attack_rate": float(y_test.mean()),
    }
    if "pointwise" in run.eval.metrics:
        m = pointwise_metrics(y_test, test_scores, threshold)
        rows.append({**base, "metric": "pointwise", **m.as_dict()})
    if "point_adjust" in run.eval.metrics:
        m = point_adjust_f1(y_test, test_scores, threshold)
        rows.append({**base, "metric": "point_adjust", **m.as_dict()})
    if "etapr" in run.eval.metrics:
        m = etapr_f1(y_test, test_scores, threshold)
        rows.append({**base, "metric": "etapr", **m.as_dict()})
    return rows


def append_summary(rows: list[dict]) -> Path:
    df_new = pd.DataFrame(rows)
    if SUMMARY_PATH.exists():
        df_old = pd.read_parquet(SUMMARY_PATH)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SUMMARY_PATH, index=False)
    return SUMMARY_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print parsed config and exit")
    args = ap.parse_args()

    run = RunConfig.from_yaml(args.config)
    print(f">> {run.name}  (hash={run.hash()})")
    print(f"   dataset={run.data.dataset}  model={run.model.name}  seeds={run.seeds}")
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
