"""Generic ARFF/CSV adapter driven by a :class:`VariantSpec`.

Unlike :mod:`morris_gas`, this adapter does not hardcode any column
conventions. It reads an ARFF/CSV, applies the variant's label / attack-id
semantics, drops the leak columns listed in the variant, masks Morris-family
sentinel values, then projects the remaining columns via
:func:`src.transfer.schema_align.project_dataframe` into the artifact's
canonical-type feature space.

Supported label semantics (in :attr:`VariantSpec.label_semantics`):
    * ``"binary_numeric"``   — column is 0/1; any non-zero clipped to 1.
    * ``"numeric_nonzero"``  — column is multiclass int; any non-zero is attack.
    * ``"binary_string"``    — column equal to ``"1"``/``"attack"`` is attack.
    * ``"string_nonnormal"`` — column not lowercase-starting-with ``"normal"`` is attack.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import read_morris_arff
from src.inference.adapters.morris_gas import AdapterResult, SchemaMismatchError
from src.inference.adapters.variants import VariantSpec
from src.transfer.schema_align import project_dataframe


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".arff":
        return read_morris_arff(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .arff or .csv.")


def _derive_labels(df: pd.DataFrame, variant: VariantSpec) -> tuple[np.ndarray | None, np.ndarray | None]:
    if variant.label_column is None or variant.label_column not in df.columns:
        return None, None

    col = df[variant.label_column]
    sem = variant.label_semantics
    if sem == "binary_numeric":
        labels = col.fillna(0).astype(np.int64).clip(0, 1).astype(np.int8)
    elif sem == "numeric_nonzero":
        labels = (col.fillna(0).astype(np.int64) != 0).astype(np.int8)
    elif sem == "binary_string":
        raw = col.astype(str).str.strip().str.lower()
        labels = raw.isin({"1", "attack", "true"}).astype(np.int8)
    elif sem == "string_nonnormal":
        raw = col.astype(str).str.strip().str.lower()
        labels = (~raw.str.startswith("normal")).astype(np.int8)
    else:
        raise ValueError(f"Unknown label_semantics '{sem}' for variant '{variant.id}'")

    if variant.attack_id_column and variant.attack_id_column in df.columns:
        ids_series = df[variant.attack_id_column].astype(str)
        attack_ids = np.where(labels.to_numpy() == 1, ids_series.to_numpy(), "normal")
    else:
        attack_ids = np.where(labels.to_numpy() == 1, "attack", "normal")

    return labels.to_numpy(), attack_ids


def load_generic_arff_file(
    path: str | Path,
    variant: VariantSpec,
    expected_features: list[str],
) -> AdapterResult:
    """Load a file under ``variant`` and project it to ``expected_features``.

    ``expected_features`` is the artifact's feature-column list; for a transfer
    artifact this is the canonical type-vector (e.g. ``["control_signal",
    "pressure", "pump_state", "setpoint", "system_state", "valve_position"]``).
    Raises :class:`SchemaMismatchError` when the variant cannot produce one
    of the expected types (missing source columns for every target type).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = _read_any(path)
    labels, attack_ids = _derive_labels(raw, variant)

    # Drop label, attack-id, and variant-declared leak columns before projection.
    drop = set(variant.drop_columns)
    if variant.label_column:
        drop.add(variant.label_column)
    if variant.attack_id_column:
        drop.add(variant.attack_id_column)
    features_raw = raw.drop(columns=[c for c in drop if c in raw.columns])

    # Morris-family sentinel masking (float32-MAX ≈ 3.4e38 used as "no reading").
    if variant.mask_sentinel_above is not None:
        thr = float(variant.mask_sentinel_above)
        numeric = features_raw.select_dtypes(include="number")
        if not numeric.empty:
            features_raw[numeric.columns] = numeric.mask(numeric.abs() > thr, 0.0)

    # Coerce feature columns to numeric (string-typed ARFF columns occasionally
    # leak in; schema_align aggregates with numpy so NaNs must be filled).
    for c in features_raw.columns:
        features_raw[c] = pd.to_numeric(features_raw[c], errors="coerce")
    features_raw = features_raw.fillna(0.0)

    # Project into canonical-type space.
    projected = project_dataframe(
        features_raw,
        feat_to_type=variant.feature_types,
        target_types=list(expected_features),
        aggregations=variant.aggregations,
    )

    # Sanity-check: did we actually get every expected column with real data?
    # A type with zero source features degrades to a zero column — that's a
    # schema mismatch worth reporting, not a silent pass.
    zero_only = [
        t for t in expected_features
        if (projected[t].to_numpy() == 0).all()
    ]
    # Only flag if the *source* side has no features for that type. If the
    # source had features but they happened to all be zero in the uploaded
    # file, that's the user's data — not a schema error.
    source_types = {
        t for t in expected_features
        if any(
            src_col in features_raw.columns
            for src_col, src_t in variant.feature_types.items()
            if src_t == t
        )
    }
    missing = [t for t in zero_only if t not in source_types]
    if missing:
        raise SchemaMismatchError(missing=missing, unexpected=[])

    return AdapterResult(features=projected, labels=labels, attack_ids=attack_ids)
