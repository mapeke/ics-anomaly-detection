"""Leave-one-process-out (LOPO) helpers for HAI.

HAI 21.03 has 4 processes (P1..P4); features are prefixed by process
and attacks are tagged in ``attack_ids`` as ``attack_P{k}``. LOPO
*drops* a process's features from the input and asks whether the
detector can still flag attacks that target that process. Unlike the
HAI<->Morris transfer, LOPO keeps the class prior fixed — it isolates
the feature-distribution-shift question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data_loader import DatasetBundle

HAI_PROCESSES = ("P1", "P2", "P3", "P4")


def drop_process_features(bundle: DatasetBundle, process: str) -> DatasetBundle:
    """Return a new :class:`DatasetBundle` with all ``{process}_*`` feature
    columns removed. Labels, splits, and attack ids are preserved."""
    if process not in HAI_PROCESSES:
        raise ValueError(f"unknown HAI process '{process}', expected one of {HAI_PROCESSES}")
    prefix = f"{process}_"
    keep = [c for c in bundle.features.columns if not c.startswith(prefix)]
    if len(keep) == len(bundle.features.columns):
        raise ValueError(
            f"No columns dropped for process '{process}'; this bundle doesn't look like HAI."
        )
    return DatasetBundle(
        features=bundle.features[keep].reset_index(drop=True).astype(np.float32),
        labels=bundle.labels,
        attack_ids=bundle.attack_ids,
        split=bundle.split,
        name=f"{bundle.name}__drop_{process}",
        timestamps=bundle.timestamps,
        metadata={**bundle.metadata, "dropped_process": process,
                  "n_features_kept": len(keep)},
    )


def f1_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 using only numpy; safe when a class is absent."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def per_attack_process_f1(
    y_true: np.ndarray, y_pred: np.ndarray, attack_ids: np.ndarray,
) -> dict[str, float]:
    """For each ``attack_P{k}`` subset, compute F1 on the union of that
    subset and all normal rows. Returns a dict keyed by the short process
    name (``'P1'`` etc.); missing processes are reported as NaN so
    downstream summary columns stay schema-stable.
    """
    out: dict[str, float] = {}
    normal_mask = (attack_ids == "normal")
    for proc in HAI_PROCESSES:
        tag = f"attack_{proc}"
        attack_mask = (attack_ids == tag)
        if not attack_mask.any():
            out[proc] = float("nan")
            continue
        mask = normal_mask | attack_mask
        out[proc] = f1_from_pred(y_true[mask], y_pred[mask])
    return out
