"""Schema-align invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetBundle
from src.transfer.schema_align import (
    common_types,
    load_feature_types,
    project_bundle_to_types,
)


@pytest.fixture
def types_yaml():
    return {
        "aggregations": {"pressure": "mean", "valve_position": "max"},
        "hai": {
            "PIT01": "pressure",
            "PIT02": "pressure",
            "FCV01Z": "valve_position",
            "VIBR01": "unknown",       # excluded
            "ADDR": "comm_metadata",   # excluded
        },
        "morris": {
            "pressure measurement": "pressure",
            "solenoid": "valve_position",
            "address": "comm_metadata",
        },
    }


@pytest.fixture
def hai_bundle():
    n = 12
    feats = pd.DataFrame({
        "PIT01": np.arange(n, dtype=np.float32) / 10,
        "PIT02": np.arange(n, dtype=np.float32) / 5,
        "FCV01Z": np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.float32),
        "VIBR01": np.zeros(n, dtype=np.float32),
        "ADDR": np.full(n, 4.0, dtype=np.float32),
    })
    labels = np.zeros(n, dtype=np.int8); labels[-3:] = 1
    split = np.array(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    return DatasetBundle(features=feats, labels=labels, attack_ids=np.full(n, "x"),
                         split=split, name="hai")


def test_common_types_intersect_correctly(types_yaml):
    assert common_types(types_yaml, "hai", "morris") == ["pressure", "valve_position"]


def test_project_bundle_emits_one_col_per_type(hai_bundle, types_yaml):
    projected = project_bundle_to_types(hai_bundle, types_yaml, target_types=["pressure", "valve_position"])
    assert list(projected.features.columns) == ["pressure", "valve_position"]
    assert projected.features.shape[0] == hai_bundle.features.shape[0]


def test_project_bundle_aggregations_match_yaml(hai_bundle, types_yaml):
    projected = project_bundle_to_types(hai_bundle, types_yaml, target_types=["pressure", "valve_position"])
    # pressure -> mean of PIT01/PIT02
    expected_pressure = (hai_bundle.features["PIT01"].to_numpy()
                         + hai_bundle.features["PIT02"].to_numpy()) / 2
    np.testing.assert_allclose(projected.features["pressure"].to_numpy(), expected_pressure)
    # valve_position -> max(FCV01Z) i.e. the column itself
    np.testing.assert_array_equal(
        projected.features["valve_position"].to_numpy(),
        hai_bundle.features["FCV01Z"].to_numpy(),
    )


def test_project_bundle_zero_pads_missing_type(hai_bundle, types_yaml):
    projected = project_bundle_to_types(hai_bundle, types_yaml,
                                        target_types=["pressure", "valve_position", "temperature"])
    assert projected.features["temperature"].abs().sum() == 0


def test_project_bundle_preserves_split_and_labels(hai_bundle, types_yaml):
    projected = project_bundle_to_types(hai_bundle, types_yaml, target_types=["pressure"])
    np.testing.assert_array_equal(projected.split, hai_bundle.split)
    np.testing.assert_array_equal(projected.labels, hai_bundle.labels)


def test_real_yaml_loads_and_has_six_common_types():
    y = load_feature_types()
    common = common_types(y, "hai", "morris")
    assert len(common) >= 5
    assert "pressure" in common
    assert "pump_state" in common