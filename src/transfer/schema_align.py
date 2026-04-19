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
    type_to_feats = _type_to_features(feat_to_type)
    if target_types is None:
        target_types = sorted(type_to_feats)

    cols = {}
    for t in target_types:
        feats = [f for f in type_to_feats.get(t, []) if f in bundle.features.columns]
        if not feats:
            # Either the dataset doesn't have this type, or features are
            # missing from the actual frame. Use zero column so the model
            # can still consume a fixed-width input.
            cols[t] = np.zeros(len(bundle.features), dtype=np.float32)
            continue
        agg = aggregations.get(t, DEFAULT_AGG)
        if agg not in _AGG_FNS:
            raise ValueError(f"unknown aggregation '{agg}' for type '{t}'")
        block = bundle.features[feats].to_numpy(dtype=np.float32)
        cols[t] = _AGG_FNS[agg](block, axis=1)

    new_features = pd.DataFrame(cols, columns=target_types).astype(np.float32)
    return DatasetBundle(
        features=new_features,
        labels=bundle.labels,
        attack_ids=bundle.attack_ids,
        split=bundle.split,
        name=f"{bundle.name}__typed",
        timestamps=bundle.timestamps,
        metadata={**bundle.metadata, "projected_from": bundle.name, "types": target_types},
    )
