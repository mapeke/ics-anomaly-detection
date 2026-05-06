"""Pydantic response models for the FastAPI routes."""
from __future__ import annotations

from pydantic import BaseModel


class ArtifactInfo(BaseModel):
    id: str                        # relative path under CHECKPOINTS_ROOT, used by /score
    model_name: str
    trained_on: str
    feature_count: int
    window: int | None
    threshold: float
    threshold_strategy: str
    config_hash: str
    seed: int
    git_sha: str


class ArtifactList(BaseModel):
    artifacts: list[ArtifactInfo]


class MetricFamily(BaseModel):
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    tap: float | None = None
    tar: float | None = None
    etapr_f1: float | None = None
    threshold: float | None = None


class PreviewRow(BaseModel):
    index: int
    score: float
    flag: int
    label: int | None = None


class ScoreResponse(BaseModel):
    artifact: ArtifactInfo
    n_input_rows: int
    n_scored: int
    n_flagged: int
    windowed: bool
    threshold: float
    metrics: dict[str, MetricFamily] | None
    preview: list[PreviewRow]
    download_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    missing: list[str] | None = None
    unexpected: list[str] | None = None
