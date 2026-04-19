# Project Plan — ICS Anomaly Detection for Cyber-Physical Security

## Context

Undergraduate diploma thesis, workshop-paper target, 4+ months from April 2026. Full plan and design rules are in [`CLAUDE.md`](CLAUDE.md); this document tracks the current state and upcoming work.

## Three thesis contributions

1. **Cross-dataset generalization** — train on HAI, test on Morris (and vice versa), plus within-HAI leave-one-process-out.
2. **Time-aware evaluation** — every number reported under point-wise, point-adjust (Xu 2018 + Kim 2022 critique), and eTaPR (Hwang 2022). Ranking changes are a finding.
3. **Per-sensor attribution** — reconstruction-error for AE family, attention rollout for transformers, precision@k against HAI attack-target labels.

## Datasets

| Dataset | Version | Status |
|---|---|---|
| HAI | 21.03 | ✅ cloned, 79 features, per-process attack flags |
| Morris Gas Pipeline | `IanArffDataset.arff` | ✅ downloaded, 16 features after leak-column removal |

`data/raw/` is gitignored; retrieval instructions in [`data/README.md`](data/README.md).

## Phase status

| Phase | Weeks | Goal | State |
|---|---|---|---|
| 1 | 1–4 | End-to-end pipeline, 4 baselines, 3 metrics, tests | ✅ **Complete** |
| 2 | 5–8 | TranAD, USAD; reproduce HAI numbers within ±0.05 F1 | ✅ **Implemented; gap documented** (TranAD HAI PA-F1 = 0.46 vs paper 0.97 on 20.07 — well outside ±0.05; expected per CLAUDE.md repro-gap policy) |
| 3 | 9–12 | HAI↔Morris transfer study + within-HAI LOPO | Not started |
| 4 | 13–16 | Attribution, thesis writing, defense | Not started |

## Phase 1 deliverables (complete)

- Config-driven experiments: `python -m experiments.run <yaml>` → one row per (seed, metric) in `results/metrics/summary.parquet`.
- 4 baselines under a common ABC: Isolation Forest, One-Class SVM, PyTorch Dense AE, PyTorch LSTM-AE.
- Three metrics with unit tests: `src/evaluation/{pointwise,point_adjust,etapr}.py`.
- Leak assertion: scaler fits on train split only and asserts no attack rows present (see `tests/test_preprocessing.py`).
- CI: ruff + pytest on push/PR; 29 tests green.
- Thin notebooks: 01/02 data-level, 03/06 read parquet, 04/05/07 placeholders.

## Reproducibility checkpoints

### Phase 1 — HAI LSTM-AE
Seed 42, val-percentile threshold=99: pointwise 0.105, PA 0.221, eTaPR 0.147.
Shin et al. 2020 report ~0.37–0.45 eTaPR for LSTM baselines on HAI 20.07. Gap attributed to dataset-version mismatch + conservative threshold + CPU-capped training.

### Phase 2 — HAI TranAD
Seed 42, GPU (RTX 3050 Ti, 4GB), 30 epochs: pointwise 0.189, PA 0.458, eTaPR 0.276.
Tuli et al. 2022 report ~0.97 PA-F1 on HAI 20.07. **Gap of ~0.5 PA-F1.** Sources:
- HAI 21.03 attack set differs from 20.07 (the working assumption).
- 30k window subsample vs. paper's full data.
- Single-layer encoder, d_model=64 vs. paper's larger config.
- Percentile-99 threshold vs. paper's oracle PA-best threshold.

Per CLAUDE.md §6, documented honestly rather than tuned to fit. Note: TranAD still ranks #1 on HAI under PA-F1 in our results (0.458 vs LSTM-AE 0.221), preserving the paper's qualitative ordering even with absolute numbers diverging.

## Phase 2 results — full 6×2 grid (eTaPR, mean over seeds)

| Model | HAI | Morris |
|---|---|---|
| Isolation Forest | 0.243 | 0.145 |
| One-Class SVM | 0.170 | 0.369 |
| Dense AE | 0.057 | 0.356 |
| LSTM-AE | 0.147 | n/a (degenerate windows) |
| USAD | 0.151 | **0.009** (model collapse on discrete Modbus) |
| TranAD | **0.276** | 0.339 |

USAD on Morris collapses (test-attack scores ≈ test-normal scores) — likely because the MLP USAD treats Morris's mostly-integer Modbus features as continuous and over-generalises. Real finding for the thesis.

## Immediate next steps

1. Run remaining Phase 1 baselines to populate `summary.parquet` fully (6 configs outstanding).
2. Execute `03_baseline_results.ipynb` and `06_metric_sensitivity.ipynb` on the populated parquet.
3. Decide compute path for Phase 2 (TranAD is ~2–4 GPU-hrs/run on HAI).
4. Advisor sign-off on the three-contribution framing (CLAUDE.md §8 open question).

## Open questions for the user

1. GPU/compute plan before Phase 2 starts.
2. Advisor approval on contributions and scope.
3. Continue targeting workshop venues or pivot to longer-form thesis-only?

## Verification (definition of done)

- `python -m experiments.run` reproduces every number in the thesis from committed YAML configs.
- `summary.parquet` has one row per (model, dataset_train, dataset_test, metric, seed) tuple, 3+ seeds per cell.
- All figures in `results/figures/` are script-generated.
- Every model class has a smoke test in CI.
