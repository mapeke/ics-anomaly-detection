"""Vanilla point-wise precision / recall / F1 / ROC-AUC / PR-AUC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class PointwiseResult:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the provided labelled set."""
    precisions, recalls, thresholds = precision_recall_curve(y, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(thresholds) == 0:
        return 0.0, 0.0
    best = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(thresholds[best]), float(f1s[best])


def pointwise_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> PointwiseResult:
    y_pred = (scores >= threshold).astype(int)
    has_both = len(np.unique(y)) > 1
    return PointwiseResult(
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        f1=f1_score(y, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y, scores) if has_both else float("nan"),
        pr_auc=average_precision_score(y, scores) if has_both else float("nan"),
        threshold=float(threshold),
    )


def plot_roc(ax, y: np.ndarray, scores: np.ndarray, label: str) -> None:
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan")
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")


def plot_pr(ax, y: np.ndarray, scores: np.ndarray, label: str) -> None:
    precisions, recalls, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores) if len(np.unique(y)) > 1 else float("nan")
    ax.plot(recalls, precisions, label=f"{label} (AP={ap:.3f})")


def plot_confusion(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
