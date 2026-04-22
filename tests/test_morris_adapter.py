"""Tests for the Morris-gas inference adapter (src/inference/adapters/morris_gas.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.adapters import SchemaMismatchError, load_morris_gas_file


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "morris.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_adapter_happy_path(tmp_path):
    df = pd.DataFrame(
        {
            "pressure": [1.0, 2.0, 3.0],
            "flow": [10.0, 20.0, 30.0],
            "binary result": ["Normal", "attack", "Normal"],
            "categorized result": ["normal", "NMRI", "normal"],
        }
    )
    out = load_morris_gas_file(_write_csv(tmp_path, df), expected_features=["pressure", "flow"])
    assert list(out.features.columns) == ["pressure", "flow"]
    assert out.features.dtypes.tolist() == [np.float32, np.float32]
    np.testing.assert_array_equal(out.labels, [0, 1, 0])
    assert out.attack_ids.tolist() == ["normal", "NMRI", "normal"]


def test_adapter_reorders_to_expected(tmp_path):
    df = pd.DataFrame(
        {"flow": [1.0, 2.0], "pressure": [10.0, 20.0], "binary result": [0, 1]}
    )
    out = load_morris_gas_file(_write_csv(tmp_path, df), expected_features=["pressure", "flow"])
    assert list(out.features.columns) == ["pressure", "flow"]
    np.testing.assert_array_equal(out.features.iloc[0].values, [10.0, 1.0])


def test_adapter_missing_column_raises(tmp_path):
    df = pd.DataFrame({"pressure": [1.0, 2.0], "binary result": [0, 1]})
    with pytest.raises(SchemaMismatchError) as exc:
        load_morris_gas_file(
            _write_csv(tmp_path, df), expected_features=["pressure", "flow"]
        )
    assert "flow" in exc.value.missing


def test_adapter_sentinel_masking(tmp_path):
    # Morris uses values near float32-MAX (~3.4e38) as "no reading"; prepare_morris_frame
    # should mask any |x| > 1e9 to 0.0.
    df = pd.DataFrame(
        {
            "pressure": [1.0, 3.4e38, 2.0],
            "flow": [10.0, 20.0, 30.0],
            "binary result": [0, 1, 0],
        }
    )
    out = load_morris_gas_file(_write_csv(tmp_path, df), expected_features=["pressure", "flow"])
    np.testing.assert_array_equal(out.features["pressure"].to_numpy(), [1.0, 0.0, 2.0])


def test_adapter_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "bad.xyz"
    p.write_text("anything")
    with pytest.raises(ValueError):
        load_morris_gas_file(str(p), expected_features=[])


def test_adapter_missing_label_column_raises(tmp_path):
    df = pd.DataFrame({"pressure": [1.0, 2.0], "flow": [10.0, 20.0]})
    with pytest.raises(KeyError):
        load_morris_gas_file(
            _write_csv(tmp_path, df), expected_features=["pressure", "flow"]
        )
