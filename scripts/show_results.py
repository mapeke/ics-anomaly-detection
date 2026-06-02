"""Pretty-print the metric grid for one run from results/metrics/summary.parquet.

Usage:
    python scripts/show_results.py baseline_morris_isolation_forest

Used in the defense demo video to read a just-completed run back out of the
results table (proves the reproducibility contract: one config -> rows on disk).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = Path("results/metrics/summary.parquet")
METRIC_ORDER = ["pointwise", "point_adjust", "etapr"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/show_results.py <run_name>")
    run_name = sys.argv[1]

    df = pd.read_parquet(SUMMARY)
    sub = df[df.run_name == run_name].copy()
    if sub.empty:
        sys.exit(f"no rows for run_name={run_name!r} in {SUMMARY}")

    # eTaPR's F1 lives in its own column; coalesce it into `f1` for display.
    sub["f1"] = np.where(sub.metric == "etapr", sub.etapr_f1, sub.f1)
    sub = sub.drop_duplicates(["seed", "metric"])

    piv = sub.pivot(index="seed", columns="metric", values="f1")
    piv = piv[[m for m in METRIC_ORDER if m in piv.columns]]

    print(f"\n  run: {run_name}")
    print(f"  config_hash: {sub.config_hash.iloc[0]}   dataset: {sub.dataset.iloc[0]}\n")
    print("  F1 by seed x metric")
    print(piv.round(3).to_string())
    print()


if __name__ == "__main__":
    main()
