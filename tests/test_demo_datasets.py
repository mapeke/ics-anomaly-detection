"""Smoke tests for the bundled demo datasets endpoint and /score wiring.

Monkeypatches DEMO_DIR alongside CHECKPOINTS_ROOT/DOWNLOADS_ROOT so the
tests don't depend on the real data/demo/ contents.
"""
from __future__ import annotations

import json

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
    demo_root = tmp_path / "demo"
    ck_root.mkdir()
    dl_root.mkdir()
    demo_root.mkdir()
    monkeypatch.setattr(routes, "CHECKPOINTS_ROOT", ck_root)
    monkeypatch.setattr(routes, "DOWNLOADS_ROOT", dl_root)
    monkeypatch.setattr(routes, "DEMO_DIR", demo_root)
    return ck_root, dl_root, demo_root


@pytest.fixture
def client(tmp_roots):
    return TestClient(create_app())


@pytest.fixture
def saved_artifact_and_demo(tmp_roots):
    ck_root, _, demo_root = tmp_roots
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
        feature_columns=feature_cols, trained_on="morris",
        config_hash="smoke", seed=42,
    )
    art_dir = ck_root / "smoke_morris_if" / "seed42"
    save_artifact(artifact, art_dir)

    # Bundled demo CSV (Morris-style label column the morris_gas adapter recognises).
    df = pd.DataFrame(X, columns=feature_cols)
    df["label"] = y
    csv_path = demo_root / "demo_sample.csv"
    df.to_csv(csv_path, index=False)

    manifest = {
        "datasets": [
            {
                "id": "demo_sample",
                "name": "Synthetic demo",
                "description": "Tiny synthetic fixture for smoke tests.",
                "n_rows": int(T),
                "n_features": int(F),
                "attack_rate": float(y.mean()),
                "source": "tests/test_demo_datasets.py",
                "filename": "demo_sample.csv",
            }
        ]
    }
    (demo_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "artifact_id": "smoke_morris_if/seed42",
        "bundled_id": "demo_sample",
    }


def test_demo_datasets_lists_manifest(client, saved_artifact_and_demo):
    r = client.get("/demo-datasets")
    assert r.status_code == 200
    body = r.json()
    assert len(body["datasets"]) == 1
    d = body["datasets"][0]
    assert d["id"] == "demo_sample"
    assert d["n_rows"] == 400
    assert d["n_features"] == 3


def test_demo_datasets_empty_when_no_manifest(client):
    """No manifest.json on disk -> empty list, not 500."""
    r = client.get("/demo-datasets")
    assert r.status_code == 200
    assert r.json() == {"datasets": []}


def test_score_with_bundled_dataset_happy_path(client, saved_artifact_and_demo):
    r = client.post(
        "/score",
        data={
            "artifact_id": saved_artifact_and_demo["artifact_id"],
            "bundled_dataset": saved_artifact_and_demo["bundled_id"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input_rows"] == 400
    assert body["n_scored"] == 400
    assert body["metrics"] is not None
    for family in ("pointwise", "point_adjust", "etapr"):
        assert family in body["metrics"]


def test_score_rejects_both_file_and_bundled(client, saved_artifact_and_demo, tmp_path):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"pressure": [1.0], "flow": [2.0], "temperature": [3.0], "label": [0]}).to_csv(csv, index=False)
    with open(csv, "rb") as f:
        r = client.post(
            "/score",
            data={
                "artifact_id": saved_artifact_and_demo["artifact_id"],
                "bundled_dataset": saved_artifact_and_demo["bundled_id"],
            },
            files={"file": ("x.csv", f, "text/csv")},
        )
    assert r.status_code == 400
    assert "exactly one" in r.json()["detail"].lower()


def test_score_rejects_neither_file_nor_bundled(client, saved_artifact_and_demo):
    r = client.post(
        "/score",
        data={"artifact_id": saved_artifact_and_demo["artifact_id"]},
    )
    assert r.status_code == 400
    assert "exactly one" in r.json()["detail"].lower()


def test_score_rejects_unknown_bundled_id(client, saved_artifact_and_demo):
    r = client.post(
        "/score",
        data={
            "artifact_id": saved_artifact_and_demo["artifact_id"],
            "bundled_dataset": "../../etc/passwd",
        },
    )
    # Manifest-id lookup never finds it -> 400, no path-construction risk.
    assert r.status_code == 400
    assert "unknown bundled_dataset" in r.json()["detail"]
