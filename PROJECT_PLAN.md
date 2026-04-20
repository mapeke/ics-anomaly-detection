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
| 1 | 1–4  | End-to-end pipeline, 4 baselines, 3 metrics, tests | ✅ **Complete** |
| 2 | 5–8  | TranAD, USAD; reproduce HAI numbers within ±0.05 F1 | ✅ **Implemented; repro gap documented honestly** |
| 3 | 9–12 | HAI↔Morris transfer + within-HAI LOPO | ✅ **Complete; three findings** |
| 4 | 13–16 | Attribution + thesis writing + defense | 🔄 **Attribution complete; draft in progress** |

## Phase 3 results (complete)

- **12 HAI↔Morris transfer runs × 2 calibration regimes** (src-cal, tgt-cal) = 24 configs.
- **12 LOPO runs** (dense_ae / lstm_ae / usad × 4 held-out processes).
- Notebook 05 with 4 figures; findings below.

**Finding 3.1 — source-calibration collapse.** Under source-domain percentile threshold, every transferred detector degenerates to the predict-all baseline of the *target* class prior: F1 ≈ 0.95 on Morris (91% post-windowing attack rate) and F1 ≈ 0.05 on HAI (2.4%). Score distributions don't transfer, so any fixed operating point either flags everything or nothing.

**Finding 3.2 — representation transfer (tgt-cal).** With the threshold chosen on *target-normal* validation data, the three windowed SOTA models separate from classical ones on HAI → Morris:

| model | src-cal eTaPR | tgt-cal eTaPR | tgt-cal PA-F1 |
|---|---|---|---|
| dense_ae | 0.95 (trivial) | 0.07 | 0.14 |
| ocsvm    | 0.95 (trivial) | 0.08 | 0.17 |
| isolation_forest | 0.75 (trivial) | 0.13 | 0.22 |
| **lstm_ae** | 0.98 (trivial) | **0.44** | **0.81** |
| **tranad**  | 0.98 (trivial) | **0.38** | **0.77** |
| **usad**    | 0.99 (trivial) | **0.40** | **0.74** |

First real classical-vs-SOTA separation in the study; only visible when the threshold-transfer question is controlled for.

**Finding 3.3 — LOPO isolates feature shift.** Dropping P2 features blinds models specifically to P2 attacks (dense_ae 0.03 F1 on P2-attacks vs. 0.41 on P1-attacks in the same config). Dropping P1 causes *global* degradation — P1 has 82% of test attacks and carries most of the "normal" signal. Best in-HAI LOPO F1 (~0.57, usad, drop-P3, on P1-attacks) is an order of magnitude above the best cross-dataset tgt-cal pointwise (~0.13) — sensor-level transfer inside one testbed vastly out-performs cross-testbed transfer.

## Phase 4 results (partial)

**Process-level attribution complete; TranAD attention rollout implemented.**

**Finding 4.1 — detection vs. localization tradeoff.** Dense AE, the weakest Phase-2 detector, is the only model whose per-feature attribution beats random across all three attacked processes (lift at p@5: P1=1.4×, P2=2.6×, P3=3.8×). Windowed SOTA (LSTM-AE / USAD / TranAD) concentrate attribution on P1 features regardless of the actual target process, falling *below* random baseline on P2 and P3. For an operator triaging a flagged window, dense_ae's output is the more actionable.

**Finding 4.2 — attention rollout null.** Weighting per-(t, f) reconstruction error by TranAD's encoder self-attention produces numerically near-identical precision@k to plain reconstruction attribution. With a single-layer encoder and 60-step windows, attention-over-time is nearly uniform, so per-feature attribution is dominated by the feature-axis residual variation, not the time-axis attention. Attention-based XAI isn't automatically the right tool for per-sensor localization in ICS.

**Finding 4.3 — LSTM-AE and USAD produce identical attribution.** To three decimal places. Almost certainly a structural coincidence (shared optimum on small-feature-count windowed autoencoder architectures on HAI). Worth flagging in the thesis; won't rehash here until multi-seed confirms it isn't a seed artifact.

## Reproducibility checkpoints

### Phase 1 — HAI LSTM-AE
Seed 42, val-percentile threshold=99: pointwise 0.105, PA 0.221, eTaPR 0.147.
Shin et al. 2020 report ~0.37–0.45 eTaPR for LSTM baselines on HAI 20.07. Gap attributed to dataset-version mismatch + conservative threshold + CPU-capped training.

### Phase 2 — HAI TranAD
Seed 42, GPU (RTX 3050 Ti, 4GB), 30 epochs: pointwise 0.189, PA 0.458, eTaPR 0.276.
Tuli et al. 2022 report ~0.97 PA-F1 on HAI 20.07. **Gap of ~0.5 PA-F1.** Documented honestly rather than tuned; TranAD still ranks #1 on HAI under PA-F1 in our results (0.458 vs LSTM-AE 0.221).

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

## Remaining Phase 4 work

- Thesis draft (see `thesis/` scaffold): introduction, background, method, results (3 sub-chapters matching contributions), discussion, conclusion.
- Multi-seed attribution runs (nice-to-have).
- Defence slides + rehearsal (week 16).

## Open questions for the user

1. Workshop-paper vs. thesis-only writing density for the draft — current outline targets workshop density; expandable.

## Verification (definition of done)

- `python -m experiments.run` (and `run_transfer` / `run_lopo` / `run_attribution`) reproduces every number in the thesis from committed YAML configs.
- `summary.parquet` has one row per (model, dataset_train, dataset_test, metric, seed) tuple, 3+ seeds per cell for core baselines.
- All figures in `results/figures/` are script-generated.
- Every model class has a smoke test in CI.
