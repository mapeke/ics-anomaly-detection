"""Process-level attribution evaluation.

HAI 21.03 labels each attack at the *process* level (``attack_P1``,
``attack_P2``, ``attack_P3``). Per-sensor ground truth is not released in
machine-readable form, so we evaluate attribution at the process level:

    precision@k = |{top-k attributed features ∩ features of the attacked process}| / k

A model whose per-feature attribution is uninformative scores at the
random baseline: ``(#features of attacked process) / (#features total)``.
A model that correctly localizes the attack to its process scores near
1.0. The gap between model precision@k and the random baseline is the
attribution signal.

The functions here are metric-only; model training and the ``attribute``
method live in ``src/models``.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def feature_to_process(feature_names: Iterable[str]) -> np.ndarray:
    """Map HAI feature names (e.g. ``P1_FCV01D``) to their process prefix
    (``'P1'``). Unrecognised names are mapped to ``'??'``.

    Returns an object ndarray of length ``len(feature_names)``.
    """
    out = []
    for n in feature_names:
        if len(n) >= 2 and n[0] == "P" and n[1].isdigit():
            out.append(n[:2])
        else:
            out.append("??")
    return np.asarray(out, dtype=object)


def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """For a 1D score vector, return a bool mask of the top-k indices."""
    if k >= len(scores):
        return np.ones_like(scores, dtype=bool)
    # argpartition on -scores gives the largest-k, unordered but correct set.
    idx = np.argpartition(-scores, k)[:k]
    mask = np.zeros_like(scores, dtype=bool)
    mask[idx] = True
    return mask


def process_precision_at_k(
    per_feature_scores: np.ndarray,
    feature_processes: np.ndarray,
    attacked_process: str,
    k: int,
) -> float:
    """``p@k`` for a single attack sample.

    Args:
        per_feature_scores: shape (F,), model's per-feature attribution
            for this sample.
        feature_processes:  shape (F,), process tag per feature (output
            of :func:`feature_to_process`).
        attacked_process:   process tag that was under attack at this sample.
        k:                  number of top features to consider.
    """
    if per_feature_scores.shape != feature_processes.shape:
        raise ValueError("score / process vectors must have the same length")
    mask = _topk_mask(per_feature_scores, k)
    topk_procs = feature_processes[mask]
    hits = int((topk_procs == attacked_process).sum())
    # Denominator is |top-k| (= F when k >= F), not k, so k > F
    # degenerates to the full-set share instead of being mis-scaled.
    return hits / max(int(mask.sum()), 1)


def precision_at_k_by_attack(
    per_feature_scores: np.ndarray,
    feature_names: Iterable[str],
    attacked_processes: np.ndarray,
    k_values: Iterable[int] = (1, 5, 10),
) -> dict[str, dict[int, float]]:
    """Aggregate p@k per attacked process across a batch of attack samples.

    Args:
        per_feature_scores: (N, F) attribution matrix, one row per sample.
        feature_names:      F-length iterable of raw HAI feature names.
        attacked_processes: (N,) object array of per-sample attack tags;
            entries outside ``{'P1','P2','P3','P4'}`` are ignored (e.g.
            'normal', 'attack' with no process disambiguation).
        k_values:           which k to report.

    Returns a nested dict: ``{process: {k: mean p@k}}``. Processes with
    zero attacks in the batch are omitted.
    """
    if per_feature_scores.ndim != 2:
        raise ValueError("per_feature_scores must be 2D (N, F)")
    feat_proc = feature_to_process(feature_names)
    N, F = per_feature_scores.shape
    if feat_proc.shape[0] != F:
        raise ValueError(f"feature_names length {feat_proc.shape[0]} != F {F}")
    if attacked_processes.shape[0] != N:
        raise ValueError("attacked_processes length must match N")

    out: dict[str, dict[int, float]] = {}
    for proc in ("P1", "P2", "P3", "P4"):
        idx = np.flatnonzero(attacked_processes == proc)
        if idx.size == 0:
            continue
        out[proc] = {}
        for k in k_values:
            vals = [
                process_precision_at_k(per_feature_scores[i], feat_proc, proc, k)
                for i in idx
            ]
            out[proc][int(k)] = float(np.mean(vals))
    return out


def random_baseline_precision(
    feature_names: Iterable[str], attacked_process: str,
) -> float:
    """Naive p@k for a uniform attribution: equals the fraction of features
    that belong to the attacked process (does not depend on k).
    """
    feat_proc = feature_to_process(feature_names)
    return float((feat_proc == attacked_process).mean())
