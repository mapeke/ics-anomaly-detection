"""HTTP routes for the external-validation app."""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.inference import load_artifact, score_dataframe
from src.inference.adapters import SchemaMismatchError, load_morris_gas_file
from src.utils import PROJECT_ROOT

from .schemas import (
    ArtifactInfo,
    ArtifactList,
    MetricFamily,
    PreviewRow,
    ScoreResponse,
)

CHECKPOINTS_ROOT = PROJECT_ROOT / "results" / "checkpoints"
DOWNLOADS_ROOT = PROJECT_ROOT / "results" / "external" / "app_runs"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

router = APIRouter()


def _discover_artifacts() -> list[ArtifactInfo]:
    if not CHECKPOINTS_ROOT.exists():
        return []
    out: list[ArtifactInfo] = []
    for manifest_path in sorted(CHECKPOINTS_ROOT.rglob("manifest.json")):
        artifact_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text())
            threshold = json.loads((artifact_dir / "threshold.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        rel = artifact_dir.relative_to(CHECKPOINTS_ROOT).as_posix()
        out.append(
            ArtifactInfo(
                id=rel,
                model_name=manifest.get("model_name", "unknown"),
                trained_on=manifest.get("trained_on", "unknown"),
                feature_count=len(manifest.get("feature_columns", [])),
                window=manifest.get("window"),
                threshold=float(threshold["value"]),
                threshold_strategy=threshold.get("strategy", "unknown"),
                config_hash=manifest.get("config_hash", ""),
                seed=int(manifest.get("seed", 0)),
                git_sha=manifest.get("git_sha", "unknown"),
            )
        )
    return out


def _resolve_artifact_dir(artifact_id: str) -> Path:
    candidate = (CHECKPOINTS_ROOT / artifact_id).resolve()
    # Path-traversal guard: resolved candidate must stay under CHECKPOINTS_ROOT.
    try:
        candidate.relative_to(CHECKPOINTS_ROOT.resolve())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="artifact_id escapes checkpoints root") from e
    if not (candidate / "manifest.json").exists():
        raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}")
    return candidate


def _json_safe(value: float | None) -> float | None:
    """JSON does not represent NaN / inf; map them to None so the UI can render ``—``."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _make_metric_family(d: dict) -> MetricFamily:
    return MetricFamily(**{k: _json_safe(v) for k, v in d.items() if k in MetricFamily.model_fields})


@router.get("/artifacts", response_model=ArtifactList)
def list_artifacts() -> ArtifactList:
    return ArtifactList(artifacts=_discover_artifacts())


@router.post("/score", response_model=ScoreResponse)
async def score(
    artifact_id: str = Form(...),
    file: UploadFile = File(...),
) -> ScoreResponse:
    artifact_dir = _resolve_artifact_dir(artifact_id)

    suffix = Path(file.filename or "upload.arff").suffix.lower() or ".arff"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        total = 0
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        artifact = load_artifact(artifact_dir)
        try:
            adapter_result = load_morris_gas_file(
                tmp_path, expected_features=artifact.feature_columns
            )
        except SchemaMismatchError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "schema_mismatch",
                    "missing": e.missing,
                    "unexpected": e.unexpected,
                },
            ) from e
        except KeyError as e:
            raise HTTPException(status_code=400, detail={"error": "missing_label_column", "detail": str(e)}) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": "unsupported_file", "detail": str(e)}) from e

        result = score_dataframe(artifact, adapter_result.features, labels=adapter_result.labels)

        run_id = uuid.uuid4().hex[:10]
        run_dir = DOWNLOADS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = run_dir / "scores.parquet"
        out_df = pd.DataFrame({"score": result.scores, "flag": result.flags.astype("int8")})
        if result.labels is not None:
            out_df["label"] = result.labels
        out_df.to_parquet(out_parquet, index=False)

        preview = [
            PreviewRow(
                index=i,
                score=float(result.scores[i]),
                flag=int(result.flags[i]),
                label=int(result.labels[i]) if result.labels is not None else None,
            )
            for i in range(min(50, len(result.scores)))
        ]

        metrics_payload: dict[str, MetricFamily] | None = None
        if result.metrics is not None:
            metrics_payload = {k: _make_metric_family(v) for k, v in result.metrics.items()}

        info = next((a for a in _discover_artifacts() if a.id == artifact_id), None)
        if info is None:
            # Should never happen — we just loaded this artifact — but stay defensive.
            raise HTTPException(status_code=500, detail="artifact vanished mid-request")

        return ScoreResponse(
            artifact=info,
            n_input_rows=result.n_input_rows,
            n_scored=int(len(result.scores)),
            n_flagged=int(result.flags.sum()),
            windowed=result.windowed,
            threshold=artifact.threshold,
            metrics=metrics_payload,
            preview=preview,
            download_url=f"/downloads/{run_id}/scores.parquet",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/downloads/{run_id}/{filename}")
def download(run_id: str, filename: str) -> FileResponse:
    # Restrict to hex-ish run ids and a tiny filename allow-list; prevents traversal.
    if not run_id.isalnum() or filename not in {"scores.parquet"}:
        raise HTTPException(status_code=404)
    path = DOWNLOADS_ROOT / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="application/octet-stream", filename=filename)
