"""Utilities: seeding, paths, logging."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in (PROCESSED_DIR, FIGURES_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and TensorFlow (if importable) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_figure(fig, name: str, subdir: str | None = None, dpi: int = 150) -> Path:
    """Save a Matplotlib figure to results/figures/[subdir/]<name>.png."""
    out_dir = FIGURES_DIR if subdir is None else FIGURES_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
