"""Cross-dataset transfer utilities.

CLAUDE.md §5.2 option 1: per-variable type tagging + score aggregation.
"""
from __future__ import annotations

from .schema_align import (
    DEFAULT_TYPES_PATH,
    common_types,
    load_feature_types,
    project_bundle_to_types,
)

__all__ = [
    "DEFAULT_TYPES_PATH",
    "common_types",
    "load_feature_types",
    "project_bundle_to_types",
]
