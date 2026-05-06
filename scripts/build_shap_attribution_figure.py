"""Render the SHAP attribution figure for Chapter 6 (Finding 4.4).

Reads the AE-family numbers from ``results/metrics/attribution.parquet``
(Phase 4) and the IF/OCSVM numbers from
``results/metrics/attribution_shap.parquet`` (Phase 5), produces a 1x3
panel of precision@5 by attacked process, with one bar per model so the
classical-vs-AE-family comparison is direct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import FIGURES_DIR, METRICS_DIR  # noqa: E402

MODEL_ORDER = (
    "isolation_forest",
    "ocsvm",
    "dense_ae",
    "lstm_ae",
    "usad",
    "tranad",
)
MODEL_LABEL = {
    "isolation_forest": "IF",
    "ocsvm": "OCSVM",
    "dense_ae": "Dense AE",
    "lstm_ae": "LSTM-AE",
    "usad": "USAD",
    "tranad": "TranAD",
}
COLOR = {
    "shap_tree": "#264653",
    "shap_kernel": "#264653",
    "reconstruction": "#2a9d8f",
    "attention": "#e9c46a",
}


def main() -> None:
    phase4_path = METRICS_DIR / "attribution.parquet"
    phase5_path = METRICS_DIR / "attribution_shap.parquet"
    if not phase5_path.exists():
        raise SystemExit(
            f"missing {phase5_path}; run scripts/build_shap_attribution.py first"
        )

    phase4 = pd.read_parquet(phase4_path)
    phase5 = pd.read_parquet(phase5_path)
    # Drop TranAD attention rows so each (model, seed, proc, k) appears once
    # in the combined frame; keep the canonical 'reconstruction' rows.
    phase4 = phase4[phase4["attribution_method"] != "attention"].copy()
    df = pd.concat([phase4, phase5], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, proc in zip(axes, ("P1", "P2", "P3")):
        sub = df[(df["attacked_process"] == proc) & (df["k"] == 5)]
        if sub.empty:
            continue
        g = sub.groupby("model")["precision_at_k"].agg(["mean", "std"])
        method_lookup = sub.drop_duplicates("model").set_index("model")["attribution_method"]
        g = g.reindex([m for m in MODEL_ORDER if m in g.index])

        x = list(range(len(g)))
        colors = [COLOR.get(method_lookup.get(m, ""), "#264653") for m in g.index]
        ax.bar(x, g["mean"], yerr=g["std"].fillna(0), capsize=4, color=colors)
        ax.axhline(
            sub["random_baseline"].iloc[0], color="#bdbdbd", linestyle="--",
            label="random baseline",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in g.index], rotation=30, ha="right")
        ax.set_title(f"attacked = {proc}")
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("precision@5")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Per-sensor attribution precision@5 (HAI test): "
        "AE-family via reconstruction error, IF/OCSVM via SHAP",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = FIGURES_DIR / "07_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_if_ocsvm_pak.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"   wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
