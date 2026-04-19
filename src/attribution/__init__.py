"""Attribution analysis for ICS anomaly detectors (CLAUDE.md §6 Phase 4).

Models in ``src/models`` already expose an ``attribute(X) -> (N, F)`` method
that returns per-feature contributions. This package evaluates whether those
contributions land on the process that was actually under attack.
"""
from __future__ import annotations

from .evaluation import (
    feature_to_process,
    precision_at_k_by_attack,
    process_precision_at_k,
)

__all__ = [
    "feature_to_process",
    "precision_at_k_by_attack",
    "process_precision_at_k",
]
