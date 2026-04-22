"""Dataset adapters for the inference pipeline.

An adapter turns a user-supplied file (ARFF/CSV/...) into the canonical
``(features_df, labels, attack_ids)`` triple, validating the feature space
against the artifact's expected columns and raising a clear
:class:`~src.inference.adapters.morris_gas.SchemaMismatchError` if the
uploaded file is not compatible with the chosen artifact.
"""
from __future__ import annotations

from .morris_gas import SchemaMismatchError, load_morris_gas_file

__all__ = ["SchemaMismatchError", "load_morris_gas_file"]
