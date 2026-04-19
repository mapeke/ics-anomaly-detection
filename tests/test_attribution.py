"""Invariants for the attribution-evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.attribution import (
    feature_to_process,
    precision_at_k_by_attack,
    process_precision_at_k,
)
from src.attribution.evaluation import random_baseline_precision


FEATS = ["P1_FCV01D", "P1_PIT01", "P2_24Vdc", "P2_OnOff",
         "P3_LCP01D", "P3_LIT01", "P4_ST_GOV", "P4_LD"]


def test_feature_to_process_tags_standard_names():
    fp = feature_to_process(FEATS)
    assert list(fp) == ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]


def test_feature_to_process_unknown_becomes_qq():
    fp = feature_to_process(["weird_name", "Attack", "X"])
    assert (fp == "??").all()


def test_process_precision_at_k_perfect_and_zero():
    feat_proc = feature_to_process(FEATS)
    scores = np.array([10., 9., 0., 0., 0., 0., 0., 0.])
    # Top-2 both P1 -> precision 1.0 against attacked P1
    assert process_precision_at_k(scores, feat_proc, "P1", k=2) == 1.0
    # Same top-2, attacked P2 -> precision 0.
    assert process_precision_at_k(scores, feat_proc, "P2", k=2) == 0.0


def test_process_precision_at_k_partial():
    feat_proc = feature_to_process(FEATS)
    # Top-3 = [P1_FCV01D, P1_PIT01, P2_24Vdc]; attacked P1 -> 2/3.
    scores = np.array([10., 9., 8., 0., 0., 0., 0., 0.])
    assert process_precision_at_k(scores, feat_proc, "P1", k=3) == pytest.approx(2 / 3)


def test_k_larger_than_F_returns_full_ratio():
    feat_proc = feature_to_process(FEATS)
    scores = np.random.default_rng(0).normal(size=len(FEATS))
    p = process_precision_at_k(scores, feat_proc, "P1", k=len(FEATS) + 5)
    # All features included in top-k, so precision = process share.
    assert p == pytest.approx(2 / len(FEATS))


def test_batch_aggregation_groups_by_process():
    N = 6
    scores = np.zeros((N, len(FEATS)))
    # Rows 0..2: attack P1 and attribution correctly points at P1 features.
    for i in range(3):
        scores[i, :2] = [10, 9]
    # Rows 3..5: attack P2 but attribution also points at P1 (wrong).
    for i in range(3, 6):
        scores[i, :2] = [10, 9]
    attacked = np.array(["P1"] * 3 + ["P2"] * 3, dtype=object)
    out = precision_at_k_by_attack(scores, FEATS, attacked, k_values=(2,))
    assert out["P1"][2] == 1.0
    assert out["P2"][2] == 0.0


def test_batch_ignores_non_process_labels():
    scores = np.random.default_rng(0).normal(size=(4, len(FEATS)))
    attacked = np.array(["P1", "normal", "attack", "P1"], dtype=object)
    out = precision_at_k_by_attack(scores, FEATS, attacked, k_values=(3,))
    # Only the two P1 rows contributed.
    assert set(out.keys()) == {"P1"}


def test_random_baseline_equals_process_share():
    assert random_baseline_precision(FEATS, "P1") == pytest.approx(2 / 8)
    assert random_baseline_precision(FEATS, "P4") == pytest.approx(2 / 8)
