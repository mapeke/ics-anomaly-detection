"""HAI 21.03 adapter.

Accepts CSV files with HAI 21.03's native sensor schema — either the raw
release files (with per-process ``attack*`` flag columns) or the cleaned
demo CSVs (with a single ``label`` column). Delegates schema normalisation
to :func:`src.data_loader.prepare_hai_frame` so it stays bit-identical to
how the training loader handled the same columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import prepare_hai_frame

from .morris_gas import AdapterResult, SchemaMismatchError


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".gz" and path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported HAI file type '{suffix}'. Use .csv or .csv.gz.")


def load_hai_file(
    path: str | Path,
    expected_features: list[str] | None = None,
) -> AdapterResult:
    """Load a HAI CSV into the canonical adapter schema.

    See :func:`src.inference.adapters.morris_gas.load_morris_gas_file` for the
    ``expected_features`` semantics — same contract.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = _read_any(path)
    prepared = prepare_hai_frame(raw)

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
