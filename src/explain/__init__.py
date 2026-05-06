"""SHAP attribution for the two classical baseline detectors.

Phase 4's per-sensor attribution work covers the four AE-family /
windowed models (Dense AE, LSTM AE, USAD, TranAD) via reconstruction
error or attention rollout, all under :mod:`src.attribution`. This
module fills the remaining gap — Isolation Forest and One-Class SVM,
where reconstruction-error attribution does not apply — using SHAP:
``shap.TreeExplainer`` for IF (exact, fast) and ``shap.KernelExplainer``
for OCSVM (approximate, slower).

For all other model types :func:`explain_row` raises
``NotImplementedError``: AE-family models should keep using
:mod:`src.attribution`.
"""
from __future__ import annotations

from .attribution import AttributionResult, explain_row

__all__ = ["AttributionResult", "explain_row"]
