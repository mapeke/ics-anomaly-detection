"""Evaluation metrics for time-series anomaly detection.

Three families, all taking (y_true, scores, threshold) -> metrics dict:
    - pointwise: vanilla per-timestep precision/recall/F1 + ROC/PR AUC.
    - point_adjust: Xu et al. 2018 protocol (inflated; Kim et al. 2022 critique).
    - etapr: enhanced time-series aware precision/recall (Hwang et al. 2022).
"""
from __future__ import annotations

from .point_adjust import point_adjust_f1
from .pointwise import (
    best_f1_threshold,
    plot_confusion,
    plot_pr,
    plot_roc,
    pointwise_metrics,
)
from .etapr import etapr_f1

__all__ = [
    "pointwise_metrics",
    "best_f1_threshold",
    "point_adjust_f1",
    "etapr_f1",
    "plot_roc",
    "plot_pr",
    "plot_confusion",
]
