# 3. Method

This chapter describes the implementation that produces every number in the thesis. All code lives in `src/` and `experiments/`; YAML configs in `experiments/configs/` are the canonical record of every run.

## 3.1 Unified detector interface

`src/models/base.py` defines an abstract `AnomalyDetector` with three methods:
- `fit(X_train, X_val)` — trained on normal data only.
- `score(X) → (N,)` — per-sample anomaly score, higher = more anomalous.
- `attribute(X) → (N, F)` — per-feature contribution; default raises, overridden by AE family and TranAD.

All six detectors implement this interface. The runner (`experiments/run.py`) treats them uniformly, so cross-dataset study, LOPO, and attribution use the same code path as same-dataset detection.

## 3.2 Data loaders (`src/data_loader.py`)

Both loaders return a `DatasetBundle`:
```
features:   (T, F) numeric DataFrame
labels:     (T,)   int8
attack_ids: (T,)   object (tag per row, e.g. "attack_P1")
split:      (T,)   "train" | "val" | "test"
name, timestamps, metadata
```

**HAI 21.03** (§3.2.1): consolidates per-process `attack_P{n}` flags into a single `label` and keeps the first-fired flag as the per-row `attack_id`. Train/val split is random over the normal `train*.csv.gz` files; test is the full attack-mixed set.

**Morris gas-pipeline** (§3.2.2): ARFF parser with a `pressure measurement` sentinel fix — values near float32-max (~3.4e38) are replaced with 0 because MinMax scaling and the schema-align mean were otherwise dominated by them.

**Leak assertion:** `DatasetBundle.assert_no_attack_in_train_val` runs before any downstream step. Attack rows in train/val are the most common source of inflated published numbers; we fail loudly.

## 3.3 Preprocessing (`src/preprocessing.py`)

- `scale_bundle(bundle) → ScaledArrays` fits `MinMaxScaler` on **train only**, transforms all splits. Assertion triggers before fit.
- `make_windows(X, window, stride)` produces `(N, W, F)` sliding windows for windowed models.
- `window_labels(y, window, stride)` tags a window as 1 iff any frame inside is anomalous.
- `percentile_threshold(val_scores, 99.0)` picks the operating point from validation scores (never from test).

## 3.4 Evaluation (`src/evaluation/`)

- `pointwise.py` — precision, recall, F1, ROC-AUC, PR-AUC.
- `point_adjust.py` — Xu-2018 PA-F1 (every frame inside an attack event is granted credit once any frame is flagged).
- `etapr.py` — Hwang-2022 event-weighted precision/recall, ported from the reference implementation and verified against their published numbers on a known model.

All three return a tiny metric class with an `.as_dict()` serializer; the runner concatenates one row per metric into `results/metrics/summary.parquet`.

## 3.5 Cross-testbed transfer (`src/transfer/schema_align.py`)

The method (CLAUDE.md §5.2 option 1) is:
1. Hand-tag every HAI and Morris feature with a canonical type (`pressure`, `pump_state`, `setpoint`, `system_state`, `valve_position`, `control_signal`, plus `unknown`/`comm_metadata` which are excluded). Tags live in `data/feature_types.yaml`.
2. For each dataset, aggregate same-typed features into a single type column (default: mean; `max` for binary-state types via the `aggregations` block). This produces a 6-column type-vector with one row per original timestep.
3. Project both datasets to the intersection of types. Train on source type-vector; score on target type-vector.

Two calibration regimes are implemented in `experiments/run_transfer.py`:
- `val_percentile` — threshold from 99th percentile of source validation scores.
- `target_val_percentile` — threshold from 99th percentile of target-normal validation scores. Decouples representation transfer from operating-point transfer.

## 3.6 Within-HAI LOPO (`src/transfer/lopo.py`)

- `drop_process_features(bundle, "P2")` returns a new bundle with all `P2_*` columns removed.
- `per_attack_process_f1(y_true, y_pred, attack_ids)` computes F1 restricted to each process's attack subset (union with normal rows).

`experiments/run_lopo.py` runs the full pipeline with a chosen process held out and emits `f1_attack_P{1..4}` columns in `summary.parquet`.

## 3.7 Attribution evaluation (`src/attribution/evaluation.py`)

- `feature_to_process(feature_names)` maps HAI feature names to their `P{n}` prefix.
- `process_precision_at_k(scores_F, feature_processes, attacked_process, k)` — fraction of top-k features that belong to the attacked process.
- `precision_at_k_by_attack` aggregates across a batch of attack windows, keyed by attacked process.

The random baseline for precision@k equals the share of features in the attacked process; we report it alongside the model numbers so lift is visible at a glance.

### 3.7.1 TranAD attention rollout

`TranADModel.attribute_attention(X)` hooks the encoder's self-attention, averages over heads and over output positions to get a per-input-timestep weight, and weights the per-(t, f) squared reconstruction error by that weight. Single-layer rollout degenerates to the attention matrix. This is a literal reading of "attention rollout as attribution" for the TranAD architecture.

## 3.8 Experiment infrastructure

- `experiments/run.py`, `run_transfer.py`, `run_lopo.py`, `run_attribution.py` — four drivers, all config-YAML-driven.
- `results/metrics/summary.parquet` (detection) and `attribution.parquet` (attribution).
- Seeding via `src.utils.set_seed(42)` at the top of every runner.
- Config-hash written into every result row so a saved number can be traced back to its exact config.

## 3.9 Tests

- `tests/test_data_loader.py` — shape invariants, leak assertion.
- `tests/test_preprocessing.py` — scaler-on-train-only, window labels.
- `tests/test_evaluation.py` — metric invariants (all-correct → 1.0; all-wrong → 0.0; sklearn agreement).
- `tests/test_transfer.py` — schema-align intersection, aggregations, zero-padding for missing types.
- `tests/test_attribution.py` — precision@k edge cases, random baseline.
- `tests/test_models_smoke.py` — a 2-epoch fit+score smoke per detector.
