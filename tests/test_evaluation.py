"""Pin metric behaviour with known-answer tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    best_f1_threshold,
    etapr_f1,
    point_adjust_f1,
    pointwise_metrics,
)
from src.evaluation.point_adjust import _expand_predictions


# ---------------------------------------------------------------------------
# Point-wise
# ---------------------------------------------------------------------------

def test_pointwise_perfect_detection():
    y = np.array([0, 0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
    m = pointwise_metrics(y, scores, threshold=0.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0


def test_pointwise_random_scores_single_class_y_yields_nan_auc():
    y = np.zeros(10, dtype=int)
    scores = np.random.default_rng(0).random(10)
    m = pointwise_metrics(y, scores, threshold=0.5)
    assert np.isnan(m.roc_auc)
    assert np.isnan(m.pr_auc)


def test_best_f1_threshold_recovers_obvious_cutoff():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    thr, f1 = best_f1_threshold(scores, y)
    assert f1 == pytest.approx(1.0, abs=1e-6)
    assert 0.3 <= thr <= 0.8


# ---------------------------------------------------------------------------
# Point-adjust
# ---------------------------------------------------------------------------

def test_point_adjust_expands_a_single_positive_to_full_window():
    y_true = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])   # one hit inside window
    adjusted = _expand_predictions(y_true, y_pred)
    np.testing.assert_array_equal(adjusted, np.array([0, 0, 1, 1, 1, 1, 0, 0]))


def test_point_adjust_leaves_missed_window_missed():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    np.testing.assert_array_equal(_expand_predictions(y_true, y_pred), y_pred)


def test_point_adjust_trailing_window():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1])
    adj = _expand_predictions(y_true, y_pred)
    np.testing.assert_array_equal(adj, np.array([0, 0, 1, 1, 1]))


def test_point_adjust_f1_beats_pointwise_f1_on_sparse_detections():
    """Sanity: PA expansion should not decrease F1 relative to point-wise."""
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    scores = np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0])
    pw = pointwise_metrics(y, scores, threshold=0.5).f1
    pa = point_adjust_f1(y, scores, threshold=0.5).f1
    assert pa >= pw


# ---------------------------------------------------------------------------
# eTaPR
# ---------------------------------------------------------------------------

def test_etapr_perfect_prediction_yields_f1_one():
    y = np.array([0, 0, 1, 1, 1, 0, 0])
    s = y.astype(float)
    r = etapr_f1(y, s, threshold=0.5)
    assert r.tap == 1.0
    assert r.tar == 1.0
    assert r.etapr_f1 == 1.0


def test_etapr_no_predictions_yields_zero_recall():
    y = np.array([0, 1, 1, 0])
    s = np.zeros_like(y, dtype=float)
    r = etapr_f1(y, s, threshold=0.5)
    assert r.tar == 0.0


def test_etapr_partial_overlap_is_between_zero_and_one():
    y = np.array([0, 1, 1, 1, 1, 0])
    s = np.array([0, 1, 1, 0, 0, 0])  # covers first half of the window
    r = etapr_f1(y, s, threshold=0.5)
    assert 0.0 < r.etapr_f1 < 1.0
    assert 0.0 < r.tar < 1.0


def test_etapr_penalises_false_positive_event():
    y = np.array([0, 0, 1, 1, 0, 0])
    s = np.array([0, 1, 1, 1, 0, 0])  # FP event at t=1 adjoining the true one
    r = etapr_f1(y, s, threshold=0.5)
    assert r.tap < 1.0
