"""Phase B tests: project_dataframe, generic_arff adapter, recalibration,
/variants endpoint, and end-to-end scoring of the gas_final variant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from app import routes
from app.main import create_app
from src.inference import (
    ModelArtifact,
    load_artifact,
    save_artifact,
    score_dataframe,
)
from src.inference.adapters import (
    SchemaMismatchError,
    get_variant,
    list_variants,
    load_generic_arff_file,
)
from src.models import build
from src.transfer import project_dataframe


# --- project_dataframe ----------------------------------------------------


def test_project_dataframe_basic():
    df = pd.DataFrame(
        {"p1": [1.0, 2.0, 3.0], "p2": [3.0, 2.0, 1.0], "v1": [0.0, 1.0, 0.0]}
    )
    feat_to_type = {"p1": "pressure", "p2": "pressure", "v1": "valve_position"}
    out = project_dataframe(
        df, feat_to_type, target_types=["pressure", "valve_position"],
        aggregations={"valve_position": "max"},
    )
    np.testing.assert_array_almost_equal(out["pressure"].to_numpy(), [2.0, 2.0, 2.0])
    np.testing.assert_array_almost_equal(out["valve_position"].to_numpy(), [0.0, 1.0, 0.0])
    assert out.dtypes.tolist() == [np.float32, np.float32]


def test_project_dataframe_missing_type_emits_zero_column():
    df = pd.DataFrame({"p1": [1.0, 2.0]})
    feat_to_type = {"p1": "pressure"}
    out = project_dataframe(
        df, feat_to_type, target_types=["pressure", "pump_state", "setpoint"]
    )
    assert list(out.columns) == ["pressure", "pump_state", "setpoint"]
    np.testing.assert_array_equal(out["pump_state"].to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(out["setpoint"].to_numpy(), [0.0, 0.0])


def test_project_dataframe_excludes_unknown_and_comm_metadata():
    df = pd.DataFrame({"p1": [1.0], "junk": [99.0], "addr": [4.0]})
    feat_to_type = {"p1": "pressure", "junk": "unknown", "addr": "comm_metadata"}
    out = project_dataframe(df, feat_to_type, target_types=["pressure"])
    assert list(out.columns) == ["pressure"]
    np.testing.assert_array_equal(out["pressure"].to_numpy(), [1.0])


# --- variant catalog ------------------------------------------------------


def test_variants_catalog_includes_morris_gas_final():
    ids = {v.id for v in list_variants()}
    assert "morris_gas_final" in ids
    v = get_variant("morris_gas_final")
    assert v.label_column == "result"
    assert v.label_semantics == "numeric_nonzero"
    assert v.feature_types["measurement"] == "pressure"


# --- generic_arff adapter -------------------------------------------------


def _make_gas_final_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "command_address": [4, 4, 4, 4],
        "response_address": [4, 4, 4, 4],
        "gain": [0.2, 0.2, 0.2, 0.2],
        "deadband": [0.5, 0.5, 0.5, 0.5],
        "setpoint": [20.0, 20.0, 20.0, 20.0],
        "control_mode": [2, 2, 2, 2],
        "control_scheme": [1, 1, 1, 1],
        "pump": [0, 0, 1, 0],
        "solenoid": [0, 1, 0, 0],
        "measurement": [0.5, 0.6, 5.0, 0.55],
        "time": [0.1, 0.2, 0.3, 0.4],
        "result": [0, 0, 6, 0],
    })
    p = tmp_path / "gas_final.csv"
    df.to_csv(p, index=False)
    return p


def test_generic_arff_projects_to_expected_types(tmp_path):
    csv = _make_gas_final_csv(tmp_path)
    variant = get_variant("morris_gas_final")
    expected = ["control_signal", "pressure", "pump_state", "setpoint", "system_state", "valve_position"]
    out = load_generic_arff_file(csv, variant=variant, expected_features=expected)
    assert list(out.features.columns) == expected
    np.testing.assert_array_equal(out.labels, [0, 0, 1, 0])
    assert out.attack_ids[2] == "6"          # multiclass id preserved
    assert out.attack_ids[0] == "normal"
    np.testing.assert_array_almost_equal(
        out.features["pressure"].to_numpy(), [0.5, 0.6, 5.0, 0.55]
    )


def test_generic_arff_schema_mismatch_when_no_columns_for_type(tmp_path):
    # Variant declares only pressure; artifact also wants 'temperature'
    # which the variant has no source columns for -> SchemaMismatchError.
    csv = _make_gas_final_csv(tmp_path)
    variant = get_variant("morris_gas_final")
    with pytest.raises(SchemaMismatchError) as exc:
        load_generic_arff_file(csv, variant=variant, expected_features=["pressure", "temperature"])
    assert "temperature" in exc.value.missing


def test_generic_arff_sentinel_masking(tmp_path):
    df = pd.DataFrame({"measurement": [1.0, 3.4e38, 2.0], "result": [0, 1, 0]})
    p = tmp_path / "with_sentinel.csv"
    df.to_csv(p, index=False)
    variant = get_variant("morris_gas_final")
    out = load_generic_arff_file(p, variant=variant, expected_features=["pressure"])
    np.testing.assert_array_almost_equal(out.features["pressure"].to_numpy(), [1.0, 0.0, 2.0])


# --- pipeline recalibration ----------------------------------------------


@pytest.fixture
def recal_inputs():
    rng = np.random.default_rng(0)
    T, F = 400, 4
    X = rng.normal(size=(T, F)).astype(np.float32)
    y = np.zeros(T, dtype=np.int8)
    X[100:130] += 6.0
    y[100:130] = 1
    cols = [f"f{i}" for i in range(F)]
    df = pd.DataFrame(X, columns=cols)
    scaler = MinMaxScaler().fit(X[:250])
    model = build("isolation_forest").fit(scaler.transform(X[:250]).astype(np.float32))
    val_scores = model.score(scaler.transform(X[250:300]).astype(np.float32))
    threshold = float(np.percentile(val_scores, 99.0))
    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=99.0,
        feature_columns=cols, trained_on="synthetic", config_hash="recal", seed=0,
    )
    return artifact, df, y


def test_recalibrate_changes_threshold(recal_inputs):
    artifact, df, y = recal_inputs
    base = score_dataframe(artifact, df, labels=y)
    recal = score_dataframe(artifact, df, labels=y, recalibrate="target_val_percentile", percentile=99.0)
    assert base.recalibrate_mode is None
    assert recal.recalibrate_mode == "target_val_percentile"
    assert recal.threshold != base.threshold        # different operating point
    assert recal.source_threshold == base.threshold  # source preserved
    # When labels are provided, recalibration uses normal-only scores; with a
    # 99th percentile of unimodal-ish noise, it should differ from the source
    # threshold derived from a separate val split.
    assert abs(recal.threshold - recal.source_threshold) > 1e-9


def test_recalibrate_no_labels_falls_back_to_all_rows(recal_inputs):
    artifact, df, _ = recal_inputs
    recal = score_dataframe(artifact, df, recalibrate="target_val_percentile", percentile=99.0)
    assert recal.recalibrate_mode == "target_val_percentile"
    assert recal.metrics is None                     # no labels -> no metrics


def test_recalibrate_unknown_mode_raises(recal_inputs):
    artifact, df, y = recal_inputs
    with pytest.raises(ValueError):
        score_dataframe(artifact, df, labels=y, recalibrate="bogus")


# --- FastAPI: /variants and /score with adapter=generic_arff -------------


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "CHECKPOINTS_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(routes, "DOWNLOADS_ROOT", tmp_path / "downloads")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "downloads").mkdir()
    return TestClient(create_app()), tmp_path


def _save_synthetic_typed_artifact(ck_root: Path) -> tuple[str, list[str]]:
    rng = np.random.default_rng(0)
    type_cols = ["control_signal", "pressure", "pump_state", "setpoint", "system_state", "valve_position"]
    X = rng.normal(size=(300, len(type_cols))).astype(np.float32)
    scaler = MinMaxScaler().fit(X[:200])
    model = build("isolation_forest").fit(scaler.transform(X[:200]).astype(np.float32))
    val_scores = model.score(scaler.transform(X[200:250]).astype(np.float32))
    threshold = float(np.percentile(val_scores, 95.0))
    artifact = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=type_cols, trained_on="synthetic_typed", config_hash="b", seed=42,
    )
    art_dir = ck_root / "synthetic_typed/seed42"
    save_artifact(artifact, art_dir)
    return "synthetic_typed/seed42", type_cols


def test_variants_endpoint(app_client):
    client, _ = app_client
    r = client.get("/variants")
    assert r.status_code == 200
    ids = {v["id"] for v in r.json()["variants"]}
    assert "morris_gas_final" in ids


def test_score_via_generic_arff(app_client, tmp_path):
    client, root = app_client
    artifact_id, _ = _save_synthetic_typed_artifact(root / "checkpoints")
    csv = _make_gas_final_csv(tmp_path)

    with open(csv, "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": artifact_id, "adapter": "generic_arff", "variant": "morris_gas_final"},
            files={"file": ("gas_final.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input_rows"] == 4
    assert body["metrics"] is not None
    assert body["recalibrate_mode"] is None


def test_score_generic_arff_missing_variant_returns_400(app_client, tmp_path):
    client, root = app_client
    artifact_id, _ = _save_synthetic_typed_artifact(root / "checkpoints")
    csv = _make_gas_final_csv(tmp_path)

    with open(csv, "rb") as f:
        r = client.post(
            "/score",
            data={"artifact_id": artifact_id, "adapter": "generic_arff"},  # no variant
            files={"file": ("gas_final.csv", f, "text/csv")},
        )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "missing_variant"


def test_score_with_recalibration_returns_both_thresholds(app_client, tmp_path):
    client, root = app_client
    artifact_id, _ = _save_synthetic_typed_artifact(root / "checkpoints")
    csv = _make_gas_final_csv(tmp_path)

    with open(csv, "rb") as f:
        r = client.post(
            "/score",
            data={
                "artifact_id": artifact_id,
                "adapter": "generic_arff",
                "variant": "morris_gas_final",
                "recalibrate": "target_val_percentile",
                "percentile": "90",
            },
            files={"file": ("gas_final.csv", f, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recalibrate_mode"] == "target_val_percentile"
    assert body["recalibrate_percentile"] == 90.0
    assert body["source_threshold"] is not None
