"""Dataset adapters for the inference pipeline.

An adapter turns a user-supplied file (ARFF/CSV/...) into the canonical
``(features_df, labels, attack_ids)`` triple, validating the feature space
against the artifact's expected columns and raising a clear
:class:`~src.inference.adapters.morris_gas.SchemaMismatchError` if the
uploaded file is not compatible with the chosen artifact.
"""
from __future__ import annotations

from pathlib import Path

from src.inference.artifact import ModelArtifact

from .hai import load_hai_file
from .morris_gas import AdapterResult, SchemaMismatchError, load_morris_gas_file

_ADAPTERS = {
    "hai": load_hai_file,
    "morris": load_morris_gas_file,
}


def load_file(
    path: str | Path,
    kind: str,
    expected_features: list[str] | None = None,
) -> AdapterResult:
    """Dispatch by dataset kind to the matching adapter."""
    if kind not in _ADAPTERS:
        raise ValueError(f"unknown dataset kind '{kind}', expected one of {sorted(_ADAPTERS)}")
    return _ADAPTERS[kind](path, expected_features=expected_features)


def expected_input_kind(artifact: ModelArtifact) -> str:
    """Which raw dataset's schema this artifact expects at inference time.

    For native artifacts (``trained_on='hai'`` / ``'morris'``) this is the
    same as ``trained_on``. For transfer artifacts (``'morris__to__hai'``),
    the deployed-on dataset is the part after ``__to__`` — that's the raw
    schema users will upload, which we then project to canonical types.
    """
    target = artifact.extra.get("target_dataset")
    if target:
        return target
    if "__to__" in artifact.trained_on:
        return artifact.trained_on.split("__to__", 1)[1]
    return artifact.trained_on


def needs_projection(artifact: ModelArtifact) -> bool:
    """True iff the artifact was trained on canonical types and the raw input
    needs to be projected via :mod:`src.transfer.schema_align` first."""
    src = artifact.extra.get("source_dataset")
    tgt = artifact.extra.get("target_dataset")
    if src and tgt and src != tgt:
        return True
    return "__to__" in artifact.trained_on


__all__ = [
    "AdapterResult",
    "SchemaMismatchError",
    "load_morris_gas_file",
    "load_hai_file",
    "load_file",
    "expected_input_kind",
    "needs_projection",
]
