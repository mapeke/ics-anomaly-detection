"""SHAP attribution evaluation for IF and OCSVM HAI baselines.

Closes the gap in Phase 4's per-sensor attribution analysis: reuses the
existing precision@k harness (``src.attribution.evaluation``) on top of
SHAP-derived per-feature contributions for the two classical baselines
that don't have reconstruction-error attribution.

Reads saved artifacts from ``results/checkpoints/baseline_hai_*/seed*``,
which the configs ``baseline_hai_isolation_forest.yaml`` and
``baseline_hai_ocsvm.yaml`` produce. Writes
``results/metrics/attribution_shap.parquet`` with rows that match the
schema of ``results/metrics/attribution.parquet`` (run_name, config_hash,
dataset, model, seed, fit_seconds=NaN, n_attack_windows, n_features,
attacked_process, k, precision_at_k, random_baseline,
n_attack_windows_process, attribution_method).

Performance: KernelExplainer is expensive, so we subsample OCSVM attack
rows per attacked process. IF uses TreeExplainer (fast) and runs on the
full attack set.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attribution import precision_at_k_by_attack  # noqa: E402
from src.attribution.evaluation import random_baseline_precision  # noqa: E402
from src.data_loader import load_hai  # noqa: E402
from src.explain import explain_row  # noqa: E402
from src.inference import load_artifact  # noqa: E402
from src.utils import METRICS_DIR, set_seed  # noqa: E402

OUT_PATH = METRICS_DIR / "attribution_shap.parquet"
K_VALUES = (1, 5, 10)
SEEDS = (42, 7, 123)
PROCS = ("P1", "P2", "P3", "P4")
# KernelExplainer is ~1-3s/row at 79 features and nsamples=128. Subsample
# attack rows per process so the full sweep finishes in a sensible window.
OCSVM_MAX_ROWS_PER_PROC = 100
IF_MAX_ROWS_PER_PROC = None  # None = no cap; TreeExplainer is fast enough.


def _hai_attack_split() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (test_features_scaled_lazily, proc_tags, feature_names) for HAI test attacks."""
    bundle = load_hai(seed=42)
    split_arr = np.asarray(bundle.split)
    test_mask = split_arr == "test"
    test_attack_ids = np.asarray(bundle.attack_ids)[test_mask]

    # Strip "attack_" prefix to land on "P1"/"P2"/"P3"/"P4"; "normal" rows fall through.
    proc_tags = np.array(
        [t.replace("attack_", "") if t.startswith("attack_") else t for t in test_attack_ids],
        dtype=object,
    )
    test_features = bundle.features.loc[test_mask].reset_index(drop=True)
    feature_names = list(test_features.columns)

    attack_mask = np.isin(proc_tags, list(PROCS))
    return test_features[attack_mask].reset_index(drop=True), proc_tags[attack_mask], feature_names


def _subsample_per_proc(
    proc_tags: np.ndarray, max_per_proc: int | None, seed: int,
) -> np.ndarray:
    """Return row indices subsampled to at most ``max_per_proc`` per process tag."""
    if max_per_proc is None:
        return np.arange(len(proc_tags))
    rng = np.random.default_rng(seed)
    keep = []
    for proc in PROCS:
        idx = np.flatnonzero(proc_tags == proc)
        if len(idx) == 0:
            continue
        if len(idx) > max_per_proc:
            idx = rng.choice(idx, size=max_per_proc, replace=False)
            idx.sort()
        keep.append(idx)
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def run_one(model_name: str, seed: int) -> list[dict]:
    """Compute SHAP attribution + precision@k for one (model, seed) pair."""
    artifact_dir = ROOT / "results" / "checkpoints" / f"baseline_hai_{model_name}" / f"seed{seed}"
    if not (artifact_dir / "manifest.json").exists():
        raise FileNotFoundError(f"missing artifact: {artifact_dir}")

    set_seed(seed)
    artifact = load_artifact(artifact_dir)
    config_hash = json.loads((artifact_dir / "manifest.json").read_text())["config_hash"]

    attack_features, attack_tags, feat_names = _hai_attack_split()
    if list(artifact.feature_columns) != feat_names:
        raise RuntimeError(
            f"artifact feature_columns disagree with current loader output for "
            f"{artifact_dir}; cannot align."
        )

    max_rows = OCSVM_MAX_ROWS_PER_PROC if model_name == "ocsvm" else IF_MAX_ROWS_PER_PROC
    keep_idx = _subsample_per_proc(attack_tags, max_rows, seed=seed)

    X_raw = attack_features.iloc[keep_idx].to_numpy(dtype=np.float32)
    X_scaled = artifact.scaler.transform(X_raw).astype(np.float32)
    tags = attack_tags[keep_idx]

    print(
        f">> {model_name} seed={seed}  attack_rows={len(X_scaled)} "
        f"(P1={int((tags=='P1').sum())} P2={int((tags=='P2').sum())} "
        f"P3={int((tags=='P3').sum())} P4={int((tags=='P4').sum())})"
    )
    t0 = time.time()
    res = explain_row(artifact_dir, X_scaled, model_name)
    elapsed = time.time() - t0
    contributions = np.asarray(res.contributions, dtype=np.float64)
    if contributions.ndim != 2 or contributions.shape[0] != len(X_scaled):
        raise RuntimeError(
            f"unexpected SHAP output shape {contributions.shape} for {len(X_scaled)} rows"
        )
    print(f"   SHAP done in {elapsed:.1f}s ({elapsed / max(len(X_scaled), 1):.3f}s/row)")

    per_proc = precision_at_k_by_attack(contributions, feat_names, tags, K_VALUES)

    base = {
        "run_name": f"baseline_hai_{model_name}__shap_attribution",
        "config_hash": config_hash,
        "dataset": "hai",
        "model": model_name,
        "seed": seed,
        "fit_seconds": float("nan"),  # we load a saved fit; no fresh fit time to report
        "n_attack_windows": int(len(X_scaled)),
        "n_features": int(X_scaled.shape[-1]),
        "attribution_method": res.method,  # 'shap_tree' or 'shap_kernel'
    }
    rows: list[dict] = []
    for proc, k_map in per_proc.items():
        n_proc = int((tags == proc).sum())
        baseline = random_baseline_precision(feat_names, proc)
        for k, val in k_map.items():
            rows.append({
                **base,
                "attacked_process": proc,
                "k": int(k),
                "precision_at_k": float(val),
                "random_baseline": float(baseline),
                "n_attack_windows_process": n_proc,
            })
            print(
                f"   {proc} k={k}: precision_at_k={val:.4f} "
                f"random={baseline:.4f} lift={val / max(baseline, 1e-9):.2f}x  "
                f"(n_proc={n_proc})"
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models", default="isolation_forest,ocsvm",
        help="comma-separated subset of {isolation_forest, ocsvm}",
    )
    ap.add_argument(
        "--seeds", default="42,7,123",
        help="comma-separated seed override (must exist on disk under "
             "results/checkpoints/baseline_hai_<model>/seed<n>)",
    )
    ap.add_argument(
        "--out", default=str(OUT_PATH),
        help="output parquet path",
    )
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_path = Path(args.out)

    all_rows: list[dict] = []
    for model_name in models:
        for seed in seeds:
            all_rows.extend(run_one(model_name, seed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    try:
        rel = out_path.resolve().relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"   wrote {len(all_rows)} rows -> {rel}")


if __name__ == "__main__":
    main()
