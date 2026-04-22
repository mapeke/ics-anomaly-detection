"""Catalog of pre-baked dataset variants the generic_arff adapter understands.

Each variant is a self-describing YAML under ``data/feature_types_variants/``
(schema documented in ``morris_gas_final.yaml``). This module provides a
thin loader + registry so the UI can offer a dropdown of known variants
instead of requiring every user to upload their own YAML.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils import PROJECT_ROOT

VARIANTS_ROOT = PROJECT_ROOT / "data" / "feature_types_variants"


@dataclass
class VariantSpec:
    """Parsed variant definition."""

    id: str                                   # file stem (e.g. "morris_gas_final")
    name: str
    description: str
    label_column: str | None
    label_semantics: str                      # see generic_arff.py
    attack_id_column: str | None
    drop_columns: list[str]
    aggregations: dict[str, str]
    feature_types: dict[str, str]
    mask_sentinel_above: float | None = 1e9

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VariantSpec":
        path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        return cls(
            id=path.stem,
            name=raw.get("name", path.stem),
            description=raw.get("description", ""),
            label_column=raw.get("label_column"),
            label_semantics=raw.get("label_semantics", "binary_numeric"),
            attack_id_column=raw.get("attack_id_column"),
            drop_columns=list(raw.get("drop_columns", [])),
            aggregations=dict(raw.get("aggregations", {})),
            feature_types=dict(raw.get("feature_types", {})),
            mask_sentinel_above=raw.get("mask_sentinel_above", 1e9),
        )


def list_variants() -> list[VariantSpec]:
    """Return every VariantSpec found under :data:`VARIANTS_ROOT`."""
    if not VARIANTS_ROOT.exists():
        return []
    return [VariantSpec.from_yaml(p) for p in sorted(VARIANTS_ROOT.glob("*.yaml"))]


def get_variant(variant_id: str) -> VariantSpec:
    """Load a variant by id (the YAML file stem)."""
    path = VARIANTS_ROOT / f"{variant_id}.yaml"
    if not path.exists():
        raise KeyError(f"Unknown variant '{variant_id}'. Available: {[v.id for v in list_variants()]}")
    return VariantSpec.from_yaml(path)
