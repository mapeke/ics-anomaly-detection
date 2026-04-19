"""Attribution evaluation driver.

Usage:
    python -m experiments.run_attribution experiments/configs/baseline_hai_lstm_ae.yaml

Trains the model (same pipeline as ``experiments.run``) and then, instead
of point-wise F1, computes **process-level precision@k** on the subset of
test windows that are labelled as attacks. Results go to
``results/metrics/attribution.parquet`` (one row per k per attacked
process per seed). HAI-only: Morris has no per-process attack tagging.
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

from experiments.run import _build_model, _WINDOWED_MODELS, _prepare_arrays  # noqa: E402
from src.attribution import precision_at_k_by_attack  # noqa: E402
from src.attribution.evaluation import random_baseline_precision  # noqa: E402
from src.config import RunConfig  # noqa: E402
from src.data_loader import load_hai  # noqa: E402
from src.preprocessing import window_labels  # noqa: E402
from src.utils import METRICS_DIR, set_seed  # noqa: E402

ATTRIBUTION_PATH = METRICS_DIR / "attribution.parquet"
K_VALUES = (1, 5, 10)


def _attack_tags_for_windows(attack_ids: np.ndarray, window: int | None,
                             stride: int) -> np.ndarray:
    """Per-window attack tag = tag of the last frame in the window.

    Matches the ``window_labels`` convention so attribution and F1 rows
    line up on the same indices.
    """
    if window is None:
        return attack_ids
    n = len(attack_ids)
    ends = np.arange(window - 1, n, stride)
    return attack_ids[ends]


def run_once(run: RunConfig, seed: int) -> list[dict]:
    set_seed(seed)
    if run.data.dataset != "hai":
        raise ValueError("attribution eval is HAI-only (per-process labels)")

    # Reuse the same scaling / windowing pipeline as experiments.run.
    X_tr, X_val, X_te, y_te, feat_names = _prepare_arrays(run)

    # Reload bundle just for attack_ids (cheap; parses CSVs from cache).
    bundle = load_hai(val_frac=run.data.val_frac, seed=run.data.seed)
    test_attack_ids = bundle.attack_ids[bundle.split == "test"]
    win_tags = _attack_tags_for_windows(
        test_attack_ids, run.data.window, run.data.stride,
    )
    # Strip the "attack_" prefix so tags are "P1"/"P2"/"P3" to match our scheme.
    proc_tags = np.array(
        [t.replace("attack_", "") if t.startswith("attack_") else t for t in win_tags],
        dtype=object,
    )

    model = _build_model(run, X_tr, seed)
    t0 = time.time()
    model.fit(X_tr, X_val)
    fit_seconds = time.time() - t0

    if not model.supports_attribution():
        raise RuntimeError(
            f"model {run.model.name} does not implement attribute()."
        )

    # Attribution only on the attack windows — normal rows would dilute
    # the per-process average.
    attack_mask = np.isin(proc_tags, ["P1", "P2", "P3", "P4"])
    X_attacks = X_te[attack_mask]
    tags_attacks = proc_tags[attack_mask]
    if X_attacks.size == 0:
        raise RuntimeError("no per-process attack windows in test set")

    # Batch over the attack windows: `attribute` can be memory-heavy so we
    # chunk it to keep VRAM in check on 4GB cards.
    attribs = []
    B = 2048
    for i in range(0, len(X_attacks), B):
        attribs.append(model.attribute(X_attacks[i : i + B]))
    attr = np.concatenate(attribs, axis=0)  # (N_attack, F)

    per_proc = precision_at_k_by_attack(attr, feat_names, tags_attacks, K_VALUES)

    base = {
        "run_name": run.name + "__attribution",
        "config_hash": run.hash(),
        "dataset": "hai",
        "model": run.model.name,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "n_attack_windows": int(len(X_attacks)),
        "n_features": int(X_attacks.shape[-1]),
    }

    rows = []
    for proc, k_map in per_proc.items():
        n_proc = int((tags_attacks == proc).sum())
        baseline = random_baseline_precision(feat_names, proc)
        for k, val in k_map.items():
            rows.append({
                **base,
                "attacked_process": proc,
                "k": k,
                "precision_at_k": val,
                "random_baseline": baseline,
                "n_attack_windows_process": n_proc,
            })
    return rows


def append_attribution(rows: list[dict]) -> Path:
    df_new = pd.DataFrame(rows)
    if ATTRIBUTION_PATH.exists():
        df_old = pd.read_parquet(ATTRIBUTION_PATH)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_parquet(ATTRIBUTION_PATH, index=False)
    return ATTRIBUTION_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()

    run = RunConfig.from_yaml(args.config)
    print(f">> {run.name}  (hash={run.hash()})  attribution")
    print(f"   dataset={run.data.dataset}  model={run.model.name}  seeds={run.seeds}")

    all_rows = []
    for seed in run.seeds:
        print(f"   seed {seed} ...")
        rows = run_once(run, seed)
        by_proc: dict[str, list[dict]] = {}
        for r in rows:
            by_proc.setdefault(r["attacked_process"], []).append(r)
        for proc in sorted(by_proc):
            bits = "  ".join(f"p@{r['k']}={r['precision_at_k']:.2f}" for r in by_proc[proc])
            first = by_proc[proc][0]
            print(f"     {proc}: {bits}  "
                  f"(random={first['random_baseline']:.2f} "
                  f"n_wins={first['n_attack_windows_process']})")
        all_rows.extend(rows)
    out = append_attribution(all_rows)
    print(f"   wrote {len(all_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
