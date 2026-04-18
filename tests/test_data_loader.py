"""DatasetBundle invariants and synthetic-bundle fixtures.

Tests that hit the real HAI/Morris files are marked `slow` and skipped
unless the data is present and `-m slow` is passed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetBundle, load_hai, load_morris
from src.utils import RAW_DIR


def _mini_bundle() -> DatasetBundle:
    n_train, n_val, n_test = 40, 10, 30
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(n_train + n_val + n_test, 4)).astype(np.float32)
    labels = np.zeros(len(feats), dtype=np.int8)
    labels[-8:] = 1
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    attack_ids = np.where(labels > 0, "synth", "normal")
    return DatasetBundle(
        features=pd.DataFrame(feats, columns=list("abcd")),
        labels=labels,
        attack_ids=attack_ids,
        split=split,
        name="synth",
    )


# ---------------------------------------------------------------------------
# Invariants on the bundle itself
# ---------------------------------------------------------------------------

def test_mask_and_x_y_helpers_consistent():
    b = _mini_bundle()
    assert b.X("train").shape == (40, 4)
    assert b.y("test").shape == (30,)
    assert int(b.y("train").sum()) == 0


def test_leak_assertion_passes_on_clean_bundle():
    _mini_bundle().assert_no_attack_in_train_val()


def test_leak_assertion_fires_when_attack_in_train():
    b = _mini_bundle()
    b.labels[0] = 1  # inject
    with pytest.raises(AssertionError):
        b.assert_no_attack_in_train_val()


def test_features_contain_no_label_columns():
    b = _mini_bundle()
    for banned in ("label", "attack_id"):
        assert banned not in b.features.columns


# ---------------------------------------------------------------------------
# Real-dataset checks (gated by dataset presence)
# ---------------------------------------------------------------------------

_HAI_AVAILABLE = (RAW_DIR / "hai" / "hai-21.03").is_dir() and any(
    (RAW_DIR / "hai" / "hai-21.03").glob("train*.csv.gz")
)
_MORRIS_AVAILABLE = (RAW_DIR / "morris" / "IanArffDataset.arff").is_file()


@pytest.mark.slow
@pytest.mark.skipif(not _HAI_AVAILABLE, reason="HAI 21.03 not present under data/raw/")
def test_load_hai_shape_and_no_leak():
    b = load_hai()
    assert b.features.shape[1] >= 70
    # Train/val splits are normal by construction.
    assert int(b.y("train").sum()) == 0
    assert int(b.y("val").sum()) == 0
    # Test has a small but nonzero attack rate.
    assert 0 < b.y("test").mean() < 0.1


@pytest.mark.slow
@pytest.mark.skipif(not _MORRIS_AVAILABLE, reason="Morris ARFF not present under data/raw/")
def test_load_morris_shape_and_no_leak():
    b = load_morris()
    assert b.features.shape[1] >= 14
    assert int(b.y("train").sum()) == 0
    assert int(b.y("val").sum()) == 0
    # All attack rows should be in test.
    assert b.y("test").sum() == int(b.labels.sum())
