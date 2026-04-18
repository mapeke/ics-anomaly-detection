# CLAUDE.md — ICS Anomaly Detection Thesis

This file is persistent context for Claude Code. Read it at the start of every session before taking action.

---

## 1. Project identity

- **Repo:** `mapeke/ics-anomaly-detection`
- **Type:** Undergraduate diploma thesis, aiming for workshop-paper publishable quality (target venues: DSN Workshop, ESORICS CPS-SPC, NDSS workshop tracks).
- **Timeline:** 4+ months from April 2026.
- **Working style:** User implements papers from scratch; prefers PyTorch. Claude Code should write production-grade research code, not toy notebooks. Keep notebooks thin — they call into `src/`, they don't contain logic.

## 2. Thesis thesis (the argument)

> Current ICS anomaly detectors report high F1 on the dataset they were trained on, but (a) their rankings change drastically under time-aware evaluation metrics, and (b) they fail to generalize across testbeds. This work quantifies both effects on HAI and Morris, benchmarks classical baselines against modern transformer-based detectors, and adds per-sensor attribution to evaluate detection *and* localization.

Three contributions, in priority order:
1. **Cross-dataset generalization study** (core novelty): train on HAI, test on Morris, and vice versa, plus leave-one-process-out within HAI.
2. **Time-aware evaluation** (rigor): every model reported under point-wise F1, point-adjust F1, and eTaPR. Ranking changes are a finding.
3. **Per-sensor attribution** (the "so what"): reconstruction-error attribution for AE family, attention-based attribution for transformers, evaluated against labeled attack targets.

Anything that doesn't serve one of these three contributions is out of scope. Push back if asked to add scope creep.

## 3. Repo layout (target state)

```
ics-anomaly-detection/
├── CLAUDE.md                     # this file
├── PROJECT_PLAN.md               # high-level plan (keep in sync with this)
├── README.md
├── requirements.txt
├── pyproject.toml                # add: ruff, pytest, mypy config
├── .gitignore
├── data/
│   ├── raw/                      # gitignored
│   ├── processed/                # parquet
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py                 # dataclass configs per experiment
│   ├── data_loader.py            # load_hai(), load_morris(), unified schema
│   ├── preprocessing.py          # scalers, sliding windows, splits
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── pointwise.py          # precision/recall/F1/ROC/PR
│   │   ├── point_adjust.py       # PA-F1 with the 2022 critique noted
│   │   └── etapr.py              # event-based eTaPR
│   ├── attribution/
│   │   ├── __init__.py
│   │   ├── reconstruction.py     # per-feature error attribution
│   │   └── attention.py          # transformer attention rollout
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # AnomalyDetector ABC: fit/score/attribute
│   │   ├── isolation_forest.py
│   │   ├── ocsvm.py
│   │   ├── autoencoder.py
│   │   ├── lstm_autoencoder.py
│   │   ├── usad.py
│   │   └── tranad.py
│   ├── transfer/
│   │   ├── __init__.py
│   │   └── schema_align.py       # handling HAI↔Morris feature mismatch
│   └── utils.py                  # seeding, paths, logging
├── experiments/
│   ├── configs/                  # YAML per run
│   └── run.py                    # CLI: python -m experiments.run <config>
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_sanity.ipynb
│   ├── 03_baseline_results.ipynb
│   ├── 04_sota_results.ipynb
│   ├── 05_cross_dataset.ipynb
│   ├── 06_metric_sensitivity.ipynb
│   └── 07_attribution.ipynb
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   └── test_models_smoke.py
└── results/
    ├── figures/
    ├── metrics/                  # parquet, not CSV, for nested cols
    └── checkpoints/              # gitignored
```

## 4. Design rules (enforce these)

- **One `AnomalyDetector` ABC.** Every model implements `fit(X_train, X_val)`, `score(X) -> per-timestep anomaly score`, and `attribute(X) -> per-feature contribution`. No exceptions. This is what makes cross-dataset and metric-sensitivity studies tractable.
- **Scaler fit on normal training data only.** Leakage here is the #1 reason published ICS numbers are inflated. Add an assertion in `preprocessing.py`.
- **Attack rows never in train/val.** Enforce in `data_loader.py`, not in notebooks.
- **Seeds everywhere.** `utils.set_seed(42)` at the top of every experiment. Record seed in results.
- **Config-driven experiments.** No hardcoded hyperparameters in notebooks. Every run is `python -m experiments.run experiments/configs/<name>.yaml` and produces a row in `results/metrics/summary.parquet` with the config hash.
- **Parquet not CSV** for results — we need nested columns (per-metric per-attack-type).
- **Tests are mandatory for `evaluation/` and `data_loader.py`.** Everything else can be smoke-tested. If Claude Code adds a new metric or loader, it writes the test in the same commit.

## 5. Critical technical decisions

### 5.1 HAI version
Use **HAI 23.05** if Git LFS budget is available on the mirror; fall back to **21.03** otherwise. Resolve the README/PROJECT_PLAN.md inconsistency by checking what actually downloads, then picking one and updating both docs. Do not leave them contradictory.

### 5.2 HAI ↔ Morris feature-space mismatch (the hardest decision)
HAI has ~80 sensors across 4 processes at 1 Hz; Morris has ~20 features from Modbus RTU at sub-second. They share no common sensor semantics. Options, in order of methodological cleanliness:

1. **Per-model transfer via score aggregation**: fit independent per-variable models on source dataset, transfer the *scoring function* (not the weights) to target dataset's variables grouped by type (pressure, flow, control signal). Requires hand-tagging variable types — one afternoon of work, documented in `data/feature_types.yaml`.
2. **Learned projection head**: train an encoder that maps both datasets into a shared latent space via contrastive loss on "normal windows." Novel but adds a month of work and a methodological defense.
3. **Punt and do leave-one-process-out within HAI only.** Weakest contribution but safest.

**Default to option 1.** Only escalate to option 2 if month-1 goes faster than expected. Option 3 is the fallback if month-3 hits a wall.

### 5.3 Metric implementation
- Point-wise F1: use sklearn, no surprises.
- Point-adjust F1: implement from Xu et al. 2018, **and** cite Kim et al. AAAI 2022 critique in the thesis. Report both but argue PA-F1 is inflated.
- eTaPR: port from the official HAI repo reference implementation; verify against their published numbers on a known model as a sanity check.

### 5.4 Models to include
- **Baselines (month 1):** Isolation Forest, OC-SVM, Dense AE, LSTM AE.
- **SOTA (month 2):** TranAD (Tuli et al. VLDB 2022), USAD (Audibert et al. KDD 2020).
- **Do not add more.** Six models × two datasets × three metrics × two transfer directions is already 72 cells in the results matrix.

## 6. Phase plan

### Phase 1 — Foundation (weeks 1–4)
Goal: end-to-end pipeline with all four baselines producing numbers under all three metrics on both datasets, matching published baselines within reasonable tolerance.

- [ ] Set up `pyproject.toml`, ruff, pytest, pre-commit.
- [ ] Implement `data_loader.py` for HAI and Morris with unified schema: `(timestamps, features_df, labels, attack_ids)`.
- [ ] Implement `preprocessing.py`: min-max scaler (fit on normal only, assert), sliding window, 70/15/15 split on normal data.
- [ ] Implement `evaluation/pointwise.py`, `evaluation/point_adjust.py`, `evaluation/etapr.py` with tests.
- [ ] Port 4 baseline models into the `AnomalyDetector` ABC.
- [ ] **Reproducibility checkpoint:** reproduce published HAI numbers for LSTM-AE within ±0.03 F1. If you can't, your pipeline is broken — stop and debug before moving on.
- [ ] First end-to-end run via `experiments/run.py`.

### Phase 2 — SOTA models (weeks 5–8)
Goal: TranAD and USAD implemented cleanly in the same ABC, reproduced on HAI.

- [ ] Re-implement TranAD from the paper in `src/models/tranad.py`. Do not vendor the reference code — read it, then write your own. Document divergences in a comment at top of the file.
- [ ] Implement USAD similarly.
- [ ] **Reproducibility checkpoint:** reproduce TranAD's reported HAI numbers within ±0.05 F1. If not, debug or document the gap — both are acceptable outcomes but must be in the thesis.
- [ ] Write `notebooks/04_sota_results.ipynb` comparing 6 models × 2 datasets × 3 metrics.

### Phase 3 — Cross-dataset study (weeks 9–12) — the core novelty
Goal: transfer results for all 6 models in both directions.

- [ ] Implement `src/transfer/schema_align.py` with option 1 (per-variable type tagging).
- [ ] Create `data/feature_types.yaml` — hand-tagged variable types for both datasets.
- [ ] Run transfer experiments: HAI→Morris and Morris→HAI for all 6 models.
- [ ] Within-HAI leave-one-process-out as a secondary transfer study.
- [ ] `notebooks/05_cross_dataset.ipynb` — this is the headline chapter of the thesis. Expect F1 drops of 20–50 pp. That's the finding.

### Phase 4 — Attribution, writing, polish (weeks 13–16)
Goal: attribution working, thesis drafted, defense ready.

- [ ] Per-feature reconstruction attribution for AE models (trivial — per-feature MSE).
- [ ] Attention rollout for TranAD.
- [ ] Evaluate attribution: precision@k of flagged features vs. labeled attack targets in HAI metadata.
- [ ] `notebooks/07_attribution.ipynb`.
- [ ] Thesis draft complete by end of week 14. Revise weeks 15–16.
- [ ] Defense slides and rehearsal week 16.

## 7. Working with Claude Code — norms

- **Start every session by running `git status` and reading this file.** Do not assume stale context.
- **Small commits, clear messages.** One feature per commit. Conventional-commits style.
- **Never commit data files, checkpoints, or anything > 10 MB.** Check `.gitignore` before `git add -A`.
- **Never run an experiment that hasn't been config-ified.** If the user asks for a quick experiment, write the YAML first, then run it.
- **Tests before notebook results.** If a notebook produces a number the thesis will cite, there must be a test that pins the underlying function.
- **When unsure about a methodological choice, stop and ask the user.** This is a thesis — reviewers will press on every choice. Silent decisions are expensive.
- **When reproducing published results, if numbers don't match within tolerance, do not "tune until they do."** Document the gap honestly and move on. Reviewers respect this; advisors catch the alternative.

## 8. Open questions for the user (resolve before Phase 3)

1. Confirm HAI version: 21.03 or 23.05. What actually downloads on your machine?
2. Confirm compute: GPU available locally, Colab, or university cluster? TranAD training on HAI takes ~2–4 GPU-hours per run.
3. Advisor check: has the advisor signed off on the three-contribution framing? Before Phase 3 starts, this must be confirmed in writing.

## 9. Definition of done (thesis-level)

- `python -m experiments.run` reproduces every number in the thesis from committed configs.
- `results/metrics/summary.parquet` has one row per (model, dataset_train, dataset_test, metric, seed) tuple with 3+ seeds per cell.
- All figures in `results/figures/` are generated by scripts in `src/`, not hand-made.
- Every model class has a smoke test that passes in CI.
- Thesis PDF cites every methodological choice back to a paper or an explicit decision in this file.
