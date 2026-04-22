"""FastAPI smoke tests for the external-validation app.

Monkeypatches the app's checkpoint and download roots to tmp_path so the
tests neither depend on nor pollute the real results/ directory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from app import routes
from app.main import create_app
from src.inference import ModelArtifact, save_artifact
from src.models import build


@pytest.fixture
def tmp_roots(tmp_path, monkeypatch):
    ck_root = tmp_path / "checkpoints"
    dl_root = tmp_path / "downloads"
    ck_root.mkdir()
    dl_root.mkdir()
    monkeypatch.setattr(routes, "CHECKPOINTS_ROOT", ck_root)
    monkeypatch.setattr(routes, "DOWNLOADS_ROOT", dl_root)
    return ck_root, dl_root


@pytest.fixture
def client(tmp_roots):
    return TestClient(create_app())


@pytest.fixture
def saved_artifact(tmp_roots):
    ck_root, _ = tmp_roots
    rng = np.random.default_rng(0)
    T, F = 400, 3
    X = rng.normal(size=(T, F)).astype(np.float32)
    y = np.zeros(T, dtype=np.int8)
    X[100:120] += 5.0
    y[100:120] = 1

    feature_cols = ["pressure", "flow", "temperature"]
    scaler = MinMaxScaler().fit(X[:250])

    model = build("isolation_forest").fit(
        scaler.transform(X[:250]).astype(np.float32)
    )
    val_scores = model.score(scaler.transform(X[250:300]).astype(np.float32))
    threshold = float(np.percentile(val_scores, 95.0))

    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=feature_cols, trained_on="synthetic",
        config_hash="smoke", seed=42,
    )
    art_dir = ck_root / "smoke_morris_if" / "seed42"
    save_artifact(artifact, art_dir)

    # Build a matching CSV fixture (with Morris-style 'binary result' label column).
    df = pd.DataFrame(X, columns=feature_cols)
    df["binary result"] = y
    fixture_csv = ck_root.parent / "fixture.csv"
    df.to_csv(fixture_csv, index=False)

    return {"artifact_id": "smoke_morris_if/seed42", "csv": fixture_csv}


def test_artifacts_empty(client):
    r = client.get("/artifacts")
    assert r.status_code == 200
    assert r.json() == {"artifacts": []}


def test_artifacts_lists_saved(client, saved_artifact):
    r = client.get("/artifacts")
    assert r.status_code == 200
    data = r.json()
    assert len(data["artifacts"]) == 1
    a = data["artifacts"][0]
    assert a["id"] == saved_artifact["artifact_id"]
    assert a["model_name"] == "isolation_forest"
    assert a["trained_on"] == "synthetic"
    assert a["feature_count"] == 3


def test_score_happy_path(client, saved_artifact):
    with open(saved_artifact["csv"], "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": saved_artifact["artifact_id"]},
            files={"file": ("morris.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input_rows"] == 400
    assert body["n_scored"] == 400
    assert body["windowed"] is False
    assert body["metrics"] is not None
    for family in ("pointwise", "point_adjust", "etapr"):
        assert family in body["metrics"]
    f1 = body["metrics"]["pointwise"]["f1"]
    assert f1 is not None and 0.0 <= f1 <= 1.0
    assert len(body["preview"]) == 50
    assert body["download_url"].startswith("/downloads/")

    # The download endpoint serves the parquet we just produced.
    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("application/octet-stream")


def test_score_schema_mismatch_returns_400(client, saved_artifact, tmp_path):
    bad = pd.DataFrame(
        {"pressure": [1.0, 2.0], "flow": [3.0, 4.0], "binary result": [0, 1]}
        # missing "temperature"
    )
    bad_csv = tmp_path / "bad.csv"
    bad.to_csv(bad_csv, index=False)
    with open(bad_csv, "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": saved_artifact["artifact_id"]},
            files={"file": ("bad.csv", f, "text/csv")},
        )
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["error"] == "schema_mismatch"
    assert "temperature" in detail["missing"]


def test_score_unknown_artifact_returns_404(client, saved_artifact):
    with open(saved_artifact["csv"], "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": "does_not_exist/seed1"},
            files={"file": ("morris.csv", f, "text/csv")},
        )
    assert r.status_code == 404


def test_score_artifact_id_traversal_rejected(client, saved_artifact):
    with open(saved_artifact["csv"], "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": "../../etc/passwd"},
            files={"file": ("morris.csv", f, "text/csv")},
        )
    # Resolves outside CHECKPOINTS_ROOT -> 400.
    assert r.status_code == 400
