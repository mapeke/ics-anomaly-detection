"""Evaluation: metrics, threshold sweep, plotting helpers."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
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
class Metrics:
    model: str
    dataset: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def best_f1_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds on a validation set; return (best_threshold, best_f1)."""
    precisions, recalls, thresholds = precision_recall_curve(y, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best = int(np.nanargmax(f1s[:-1])) if len(thresholds) else 0
    return float(thresholds[best]), float(f1s[best])


def compute_metrics(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    model: str,
    dataset: str,
) -> Metrics:
    y_pred = (scores >= threshold).astype(int)
    return Metrics(
        model=model,
        dataset=dataset,
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        f1=f1_score(y, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y, scores) if len(np.unique(y)) > 1 else float("nan"),
        pr_auc=average_precision_score(y, scores) if len(np.unique(y)) > 1 else float("nan"),
        threshold=float(threshold),
    )


def plot_roc(ax, y: np.ndarray, scores: np.ndarray, label: str) -> None:
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")


def plot_pr(ax, y: np.ndarray, scores: np.ndarray, label: str) -> None:
    precisions, recalls, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    ax.plot(recalls, precisions, label=f"{label} (AP={ap:.3f})")


def plot_confusion(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def save_metrics_row(m: Metrics, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{m.model}_{m.dataset}.csv"
    pd.DataFrame([m.as_dict()]).to_csv(path, index=False)
    return path


def aggregate_summary(metrics_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(metrics_dir.glob("*.csv")):
        if p.name == "summary.csv":
            continue
        rows.append(pd.read_csv(p))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(metrics_dir / "summary.csv", index=False)
    return df
