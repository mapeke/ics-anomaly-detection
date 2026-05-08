"""Tests for the HAI inference adapter (src/inference/adapters/hai.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.adapters import SchemaMismatchError, load_hai_file


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "hai.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_demo_csv_with_label_only(tmp_path):
    """Cleaned demo CSV has a single 'label' column — no per-process flags."""
    df = pd.DataFrame(
        {
            "P1_PIT01": [0.1, 0.2, 0.3],
            "P2_SIT01": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        }
    )
    out = load_hai_file(_write_csv(tmp_path, df))
    assert list(out.features.columns) == ["P1_PIT01", "P2_SIT01"]
    np.testing.assert_array_equal(out.labels, [0, 1, 0])
    assert out.attack_ids.tolist() == ["normal", "attack", "normal"]


def test_per_process_attack_flags(tmp_path):
    """Raw HAI release files have per-process attack* columns."""
    df = pd.DataFrame(
        {
            "P1_PIT01": [0.1, 0.2, 0.3],
            "attack": [0, 1, 0],
            "attack_P1": [0, 1, 0],
            "attack_P2": [0, 0, 0],
        }
    )
    out = load_hai_file(_write_csv(tmp_path, df))
    assert list(out.features.columns) == ["P1_PIT01"]
    np.testing.assert_array_equal(out.labels, [0, 1, 0])
    # First triggered per-process flag becomes the attack id.
    assert out.attack_ids[1] == "attack_P1"
    assert out.attack_ids[0] == "normal"


def test_expected_features_reorders(tmp_path):
    df = pd.DataFrame(
        {"P2_SIT01": [1.0, 2.0], "P1_PIT01": [0.1, 0.2], "label": [0, 1]}
    )
    out = load_hai_file(
        _write_csv(tmp_path, df), expected_features=["P1_PIT01", "P2_SIT01"]
    )
    assert list(out.features.columns) == ["P1_PIT01", "P2_SIT01"]
    np.testing.assert_allclose(out.features.iloc[0].values, [0.1, 1.0], rtol=1e-6)


def test_missing_expected_column_raises(tmp_path):
    df = pd.DataFrame({"P1_PIT01": [0.1], "label": [0]})
    with pytest.raises(SchemaMismatchError) as exc:
        load_hai_file(
            _write_csv(tmp_path, df), expected_features=["P1_PIT01", "P2_SIT01"]
        )
    assert "P2_SIT01" in exc.value.missing


def test_no_label_column_raises(tmp_path):
    df = pd.DataFrame({"P1_PIT01": [0.1, 0.2]})
    with pytest.raises(KeyError):
        load_hai_file(_write_csv(tmp_path, df))


def test_unsupported_suffix_rejected(tmp_path):
    p = tmp_path / "bad.xyz"
    p.write_text("anything")
    with pytest.raises(ValueError):
        load_hai_file(str(p))
