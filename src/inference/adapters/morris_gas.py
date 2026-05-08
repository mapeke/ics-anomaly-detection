"""Morris gas-pipeline adapter.

Accepts ARFF or CSV files in the Morris gas-pipeline family (``IanArffDataset.arff``
variants, re-captures, etc.). Delegates the schema normalisation to
:func:`src.data_loader.prepare_morris_frame` so we stay bit-identical to how
the training loader handled the same columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import prepare_morris_frame, read_morris_arff


class SchemaMismatchError(ValueError):
    """Raised when an uploaded file's feature columns disagree with the artifact's."""

    def __init__(self, *, missing: list[str], unexpected: list[str]):
        self.missing = missing
        self.unexpected = unexpected
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        super().__init__("; ".join(parts) or "schema mismatch")


@dataclass
class AdapterResult:
    """Canonical inference input."""

    features: pd.DataFrame          # columns == expected_features, dtype float32
    labels: np.ndarray | None       # (N,) int8 or None if dataset has no labels
    attack_ids: np.ndarray | None   # (N,) str or None


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".arff":
        return read_morris_arff(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported Morris-gas file type '{suffix}'. Use .arff or .csv.")


def load_morris_gas_file(
    path: str | Path,
    expected_features: list[str] | None = None,
) -> AdapterResult:
    """Load a Morris gas-pipeline file into the canonical adapter schema.

    When ``expected_features`` is provided, validates that every expected
    column is present and reorders to match. When ``None``, returns all clean
    feature columns in source order (the routes layer then validates after
    any optional projection step). Missing-column errors raise
    :class:`SchemaMismatchError`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = _read_any(path)
    prepared = prepare_morris_frame(raw)

    actual = [c for c in prepared.columns if c not in ("label", "attack_id")]
    if expected_features is not None:
        missing = [c for c in expected_features if c not in actual]
        unexpected = [c for c in actual if c not in expected_features]
        if missing:
            raise SchemaMismatchError(missing=missing, unexpected=unexpected)
        feature_cols = list(expected_features)
    else:
        feature_cols = actual

    features = prepared[feature_cols].astype(np.float32).reset_index(drop=True)

    labels = prepared["label"].to_numpy(dtype=np.int8) if "label" in prepared.columns else None
    attack_ids = (
        prepared["attack_id"].to_numpy() if "attack_id" in prepared.columns else None
    )
    return AdapterResult(features=features, labels=labels, attack_ids=attack_ids)
