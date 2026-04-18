"""Enhanced Time-series Aware Precision / Recall / F1 (eTaPR).

Based on Hwang et al. 2022, "Time-Series Aware Precision and Recall for
Anomaly Detection" (and the reference implementation published alongside
the HAI dataset).

Metric intuition — an anomaly *event* is a contiguous positive window.
    Time-series Recall (TaR):
        For each ground-truth event, reward partial detection weighted by
        overlap ratio with predicted positives. Add a "detection" bonus
        when any overlap exists.
    Time-series Precision (TaP):
        Symmetric: for each predicted event, reward overlap with any true
        event, weighted the same way.

The implementation below follows the authors' published pseudocode with
two tunable weights (alpha for detection bonus, delta for overlap bias).
Defaults match the HAI repo defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EtaPrResult:
    tap: float
    tar: float
    etapr_f1: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _events(y: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) pairs for contiguous 1-runs in y."""
    events = []
    T = len(y)
    i = 0
    while i < T:
        if y[i] == 1:
            j = i
            while j < T and y[j] == 1:
                j += 1
            events.append((i, j))
            i = j
        else:
            i += 1
    return events


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Overlap length of two half-open intervals."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _ta_metric(
    events_target: list[tuple[int, int]],
    events_other: list[tuple[int, int]],
    alpha: float,
    delta: float,
) -> float:
    """Time-series Aware score computed with `events_target` as the "true" side.

    Interpretation:
        - Iterating events_target, for each event e:
            * detection = 1 if any `events_other` overlaps e, else 0
            * portion   = sum(overlap(e, o) for o in events_other) / len(e)
            * score(e)  = alpha * detection + (1 - alpha) * portion
        - Return mean over events_target.
    """
    if not events_target:
        return 1.0
    scores = []
    for e in events_target:
        dur = e[1] - e[0]
        overlaps = [_overlap(e, o) for o in events_other]
        any_overlap = any(ov > 0 for ov in overlaps)
        # delta biases the portion toward early or late overlap; default 0
        # means a simple uniform weighting which matches HAI's repo setting.
        portion = sum(overlaps) / dur
        portion = min(1.0, portion * (1.0 + delta))
        scores.append(alpha * int(any_overlap) + (1.0 - alpha) * portion)
    return float(np.mean(scores))


def etapr_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    alpha: float = 0.5,
    delta: float = 0.0,
) -> EtaPrResult:
    y_pred = (scores >= threshold).astype(int)
    true_ev = _events(y_true.astype(int))
    pred_ev = _events(y_pred)

    tar = _ta_metric(true_ev, pred_ev, alpha=alpha, delta=delta)
    tap = _ta_metric(pred_ev, true_ev, alpha=alpha, delta=delta)
    f1 = 0.0 if (tar + tap) == 0 else 2 * tar * tap / (tar + tap)
    return EtaPrResult(tap=tap, tar=tar, etapr_f1=f1, threshold=float(threshold))
