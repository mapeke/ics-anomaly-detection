"""Schema alignment for cross-dataset transfer.

We collapse each dataset's feature matrix to a *type-vector* — one
column per canonical type, computed by aggregating the per-row values
of all same-typed features. The aggregation function defaults to mean
but can be overridden per-type in ``data/feature_types.yaml``
(``aggregations:`` section).

A model trained on the source dataset's type-vector can then be evaluated
directly on the target dataset's type-vector because both have the same
columns (the type intersection of the two datasets).

This is the simplest defensible scheme for HAI ↔ Morris transfer; richer
options (learned projections, per-variable transfer) are out of scope
for the Phase-3 milestone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..data_loader import DatasetBundle
from ..utils import PROJECT_ROOT

DEFAULT_TYPES_PATH = PROJECT_ROOT / "data" / "feature_types.yaml"

EXCLUDE_TYPES = {"unknown", "comm_metadata"}   # types we never include in transfer
DEFAULT_AGG = "mean"
_AGG_FNS = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}


def load_feature_types(path: str | Path = DEFAULT_TYPES_PATH) -> dict:
    """Return the parsed feature_types.yaml, including ``aggregations:``."""
    return yaml.safe_load(Path(path).read_text())


def _type_to_features(types_for_dataset: dict[str, str]) -> dict[str, list[str]]:
    """Invert {feature: type} to {type: [features]}, ignoring excluded types."""
    out: dict[str, list[str]] = {}
    for feat, t in types_for_dataset.items():
        if t in EXCLUDE_TYPES:
            continue
        out.setdefault(t, []).append(feat)
    return out


def common_types(types_yaml: dict, source: str, target: str) -> list[str]:
    """Sorted intersection of types present (after exclusions) in both datasets."""
    src = set(_type_to_features(types_yaml[source]))
    tgt = set(_type_to_features(types_yaml[target]))
    return sorted(src & tgt)


def project_dataframe(
    df: pd.DataFrame,
    feat_to_type: dict[str, str],
    target_types: list[str] | None = None,
    aggregations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Collapse ``df`` (raw features) to one column per canonical type.

    Pure DataFrame-in, DataFrame-out helper reused by
    :func:`project_bundle_to_types` (bundle-wrapped training path) and by
    the inference adapters in :mod:`src.inference.adapters`. Missing types
    get a zero column so the model always sees a fixed-width input.
    """
    aggregations = aggregations or {}
    type_to_feats = _type_to_features(feat_to_type)
    if target_types is None:
        target_types = sorted(type_to_feats)

    cols: dict[str, np.ndarray] = {}
    for t in target_types:
        feats = [f for f in type_to_feats.get(t, []) if f in df.columns]
        if not feats:
            cols[t] = np.zeros(len(df), dtype=np.float32)
            continue
        agg = aggregations.get(t, DEFAULT_AGG)
        if agg not in _AGG_FNS:
            raise ValueError(f"unknown aggregation '{agg}' for type '{t}'")
        block = df[feats].to_numpy(dtype=np.float32)
        cols[t] = _AGG_FNS[agg](block, axis=1)

    return pd.DataFrame(cols, columns=target_types).astype(np.float32)


def project_bundle_to_types(
    bundle: DatasetBundle,
    types_yaml: dict | None = None,
    target_types: list[str] | None = None,
    dataset_key: str | None = None,
) -> DatasetBundle:
    """Return a new :class:`DatasetBundle` whose feature matrix has one
    column per canonical type, restricted to ``target_types`` (defaults
    to *all* types present in this dataset).

    Aggregation per type follows ``types_yaml['aggregations']``.
    """
    if types_yaml is None:
        types_yaml = load_feature_types()
    if dataset_key is None:
        dataset_key = bundle.name
    if dataset_key not in types_yaml:
        raise KeyError(f"feature_types.yaml has no entry for dataset '{dataset_key}'")

    feat_to_type = types_yaml[dataset_key]
    aggregations = types_yaml.get("aggregations", {})
    if target_types is None:
        target_types = sorted(_type_to_features(feat_to_type))

    new_features = project_dataframe(
        bundle.features, feat_to_type, target_types, aggregations=aggregations
    )

    return DatasetBundle(
        features=new_features,
        labels=bundle.labels,
        attack_ids=bundle.attack_ids,
        split=bundle.split,
        name=f"{bundle.name}__typed",
        timestamps=bundle.timestamps,
        metadata={**bundle.metadata, "projected_from": bundle.name, "types": target_types},
    )
