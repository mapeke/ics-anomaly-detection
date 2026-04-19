"""Cross-dataset transfer utilities.

CLAUDE.md §5.2 option 1: per-variable type tagging + score aggregation,
plus within-HAI leave-one-process-out as a feature-shift control.
"""
from __future__ import annotations

from .lopo import (
    HAI_PROCESSES,
    drop_process_features,
    per_attack_process_f1,
)
from .schema_align import (
    DEFAULT_TYPES_PATH,
    common_types,
    load_feature_types,
    project_bundle_to_types,
)

__all__ = [
    "DEFAULT_TYPES_PATH",
    "HAI_PROCESSES",
    "common_types",
    "drop_process_features",
    "load_feature_types",
    "per_attack_process_f1",
    "project_bundle_to_types",
]
