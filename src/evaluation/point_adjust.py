"""Point-adjust F1 (Xu et al. 2018).

Protocol: if any single timestep inside a contiguous attack window is flagged
as anomalous, the *entire* window is counted as correctly detected. This is
the default metric in most published ICS AD results and the reason those
numbers are almost unreachably high on real deployments.

Kim et al. AAAI 2022 ("Towards a rigorous evaluation of time-series anomaly
detection") show that even random scores achieve ~98% F1 under PA on many
benchmarks. We report PA-F1 **and** flag this inflation in the thesis.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .pointwise import PointwiseResult


def _expand_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Apply the point-adjust expansion: flip every timestep of any attack
    window in which at least one prediction is positive.
    """
    out = y_pred.copy()
    T = len(y_true)
    in_attack = False
    start = 0
    for t in range(T):
        if y_true[t] == 1 and not in_attack:
            in_attack = True
            start = t
        elif y_true[t] == 0 and in_attack:
            # Window ran from [start, t); check for any positive prediction.
            if y_pred[start:t].any():
                out[start:t] = 1
            in_attack = False
    if in_attack:
        if y_pred[start:T].any():
            out[start:T] = 1
    return out


def point_adjust_f1(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> PointwiseResult:
    y_pred = (scores >= threshold).astype(int)
    y_adj = _expand_predictions(y_true.astype(int), y_pred)
    return PointwiseResult(
        precision=precision_score(y_true, y_adj, zero_division=0),
        recall=recall_score(y_true, y_adj, zero_division=0),
        f1=f1_score(y_true, y_adj, zero_division=0),
        roc_auc=float("nan"),   # PA is threshold-specific; AUCs aren't well-defined.
        pr_auc=float("nan"),
        threshold=float(threshold),
    )
