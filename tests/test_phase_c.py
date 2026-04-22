"""Phase C tests: electrical vocabulary + ad-hoc variant YAML upload."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.preprocessing import MinMaxScaler

from app import routes
from app.main import create_app
from src.inference import ModelArtifact, save_artifact, score_dataframe
from src.inference.adapters import (
    VariantSpec,
    list_variants,
    load_generic_arff_file,
)
from src.models import build
from src.transfer import load_feature_types, project_dataframe


# --- C1: electrical vocabulary ------------------------------------------


def test_electrical_vocabulary_available():
    types = load_feature_types()
    aggs = types["aggregations"]
    # New electrical-family entries — regression guard.
    for key, expected_agg in {
        "voltage": "mean",
        "current": "mean",
        "voltage_angle": "mean",
        "current_angle": "mean",
        "phase_angle": "mean",
        "frequency": "mean",
        "relay_state": "max",
        "power": "mean",
        "impedance": "mean",
    }.items():
        assert key in aggs, f"missing {key} in aggregations"
        assert aggs[key] == expected_agg, f"{key} expected {expected_agg}, got {aggs[key]}"
    # Gas-family entries preserved.
    for gas_key in ("pressure", "pump_state", "setpoint", "valve_position", "control_signal"):
        assert gas_key in aggs


def test_morris_power_template_hidden_from_variants_endpoint():
    # Template lives under _templates/ — list_variants globs *.yaml at root only.
    ids = {v.id for v in list_variants()}
    assert "morris_power" not in ids
    assert "morris_gas_final" in ids   # real variant still there


# --- C3: VariantSpec.from_yaml_text -------------------------------------


SAMPLE_VARIANT_YAML = """
name: synthetic
description: synthetic
label_column: label
label_semantics: binary_numeric
feature_types:
  v1: voltage
  v2: voltage
  i1: current
  r1: relay_state
aggregations:
  voltage: mean
  current: mean
  relay_state: max
"""


def test_variant_from_yaml_text_roundtrip():
    spec = VariantSpec.from_yaml_text(SAMPLE_VARIANT_YAML)
    assert spec.id == "uploaded"
    assert spec.label_column == "label"
    assert spec.label_semantics == "binary_numeric"
    assert spec.feature_types["r1"] == "relay_state"
    assert spec.aggregations["relay_state"] == "max"

    # Project a synthetic frame through it and check canonical columns.
    df = pd.DataFrame(
        {"v1": [1.0, 2.0], "v2": [3.0, 4.0], "i1": [0.1, 0.2], "r1": [0.0, 1.0]}
    )
    projected = project_dataframe(
        df, feat_to_type=spec.feature_types,
        target_types=["voltage", "current", "relay_state"],
        aggregations=spec.aggregations,
    )
    np.testing.assert_array_almost_equal(projected["voltage"].to_numpy(), [2.0, 3.0])
    np.testing.assert_array_almost_equal(projected["current"].to_numpy(), [0.1, 0.2])
    np.testing.assert_array_almost_equal(projected["relay_state"].to_numpy(), [0.0, 1.0])


def test_variant_from_yaml_text_rejects_non_mapping():
    with pytest.raises(ValueError, match="mapping"):
        VariantSpec.from_yaml_text("- just\n- a\n- list\n")


# --- C4: FastAPI /score accepts variant_yaml upload ----------------------


def _synthetic_typed_artifact(ck_root: Path) -> tuple[str, list[str]]:
    rng = np.random.default_rng(0)
    types = ["current", "relay_state", "voltage"]
    X = rng.normal(size=(300, len(types))).astype(np.float32)
    scaler = MinMaxScaler().fit(X[:200])
    model = build("isolation_forest").fit(scaler.transform(X[:200]).astype(np.float32))
    val = model.score(scaler.transform(X[200:250]).astype(np.float32))
    threshold = float(np.percentile(val, 95.0))
    art = ModelArtifact(
        model=model, scaler=scaler, threshold=threshold,
        threshold_strategy="val_percentile", threshold_percentile=95.0,
        feature_columns=types, trained_on="synthetic_electrical", config_hash="c", seed=42,
    )
    art_dir = ck_root / "synth_electrical/seed42"
    save_artifact(art, art_dir)
    return "synth_electrical/seed42", types


def _synthetic_power_csv(tmp_path: Path) -> Path:
    # Matches SAMPLE_VARIANT_YAML column names.
    df = pd.DataFrame({
        "v1": [0.5, 0.6, 5.0, 0.55],
        "v2": [0.4, 0.5, 5.1, 0.45],
        "i1": [0.01, 0.02, 1.5, 0.015],
        "r1": [0, 0, 1, 0],
        "label": [0, 0, 1, 0],
    })
    p = tmp_path / "synth_power.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "CHECKPOINTS_ROOT", tmp_path / "checkpoints")
    monkeypatch.setattr(routes, "DOWNLOADS_ROOT", tmp_path / "downloads")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "downloads").mkdir()
    return TestClient(create_app()), tmp_path


def test_generic_arff_with_uploaded_yaml(app_client, tmp_path):
    client, root = app_client
    artifact_id, _ = _synthetic_typed_artifact(root / "checkpoints")
    csv = _synthetic_power_csv(tmp_path)

    with open(csv, "rb") as f_data:
        r = client.post(
            "/score",
            data={"artifact_id": artifact_id, "adapter": "generic_arff"},
            files={
                "file": ("synth_power.csv", f_data, "text/csv"),
                "variant_yaml": ("uploaded.yaml", SAMPLE_VARIANT_YAML.encode(), "text/yaml"),
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_input_rows"] == 4
    assert body["metrics"] is not None


def test_generic_arff_malformed_yaml_returns_400(app_client, tmp_path):
    client, root = app_client
    artifact_id, _ = _synthetic_typed_artifact(root / "checkpoints")
    csv = _synthetic_power_csv(tmp_path)

    malformed = b": : :\n  [unbalanced"
    with open(csv, "rb") as f_data:
        r = client.post(
            "/score",
            data={"artifact_id": artifact_id, "adapter": "generic_arff"},
            files={
                "file": ("synth_power.csv", f_data, "text/csv"),
                "variant_yaml": ("bad.yaml", malformed, "text/yaml"),
            },
        )
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "bad_variant_yaml"


def test_generic_arff_uploaded_yaml_wins_over_dropdown(app_client, tmp_path):
    """When both variant and variant_yaml are present, the upload wins.

    Smoke-level: if uploaded YAML describes columns the dropdown variant
    doesn't, and the request succeeds, the upload must have been applied.
    """
    client, root = app_client
    artifact_id, _ = _synthetic_typed_artifact(root / "checkpoints")
    csv = _synthetic_power_csv(tmp_path)

    with open(csv, "rb") as f_data:
        r = client.post(
            "/score",
            data={
                "artifact_id": artifact_id,
                "adapter": "generic_arff",
                "variant": "morris_gas_final",   # would fail against electrical columns
            },
            files={
                "file": ("synth_power.csv", f_data, "text/csv"),
                "variant_yaml": ("uploaded.yaml", SAMPLE_VARIANT_YAML.encode(), "text/yaml"),
            },
        )
    assert r.status_code == 200, r.text


# --- C5: CLI-level variant/variant_yaml equivalence ----------------------


def test_variant_yaml_equivalent_to_filed_variant(tmp_path):
    """Parsing the same YAML via from_yaml_text and from_yaml yields
    specs that produce identical projections on the same input."""
    yaml_path = tmp_path / "mirror.yaml"
    yaml_path.write_text(SAMPLE_VARIANT_YAML)

    spec_text = VariantSpec.from_yaml_text(SAMPLE_VARIANT_YAML)
    spec_file = VariantSpec.from_yaml(yaml_path)

    df = pd.DataFrame(
        {"v1": [0.1, 0.2], "v2": [0.3, 0.4], "i1": [0.01, 0.02], "r1": [0.0, 1.0]}
    )
    proj_text = project_dataframe(
        df, spec_text.feature_types, target_types=["voltage", "current", "relay_state"],
        aggregations=spec_text.aggregations,
    )
    proj_file = project_dataframe(
        df, spec_file.feature_types, target_types=["voltage", "current", "relay_state"],
        aggregations=spec_file.aggregations,
    )
    pd.testing.assert_frame_equal(proj_text, proj_file)
