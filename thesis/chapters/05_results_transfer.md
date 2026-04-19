# 5. Results II — Cross-testbed transfer and within-HAI LOPO

This is the headline chapter of the thesis (cf. CLAUDE.md §6 Phase 3).

Notebook: `05_cross_dataset.ipynb`.
Figures: F2, F3, F4 in `results/figures/05_cross_dataset/`.
Source rows: `summary.parquet` rows with `dataset_train != dataset_test` (HAI↔Morris) and `dataset == 'hai__lopo'` (LOPO).

## 5.1 Schema alignment

Six canonical types intersect HAI and Morris after hand-tagging in `data/feature_types.yaml`: pressure, pump_state, setpoint, system_state, valve_position, control_signal. HAI contributes 79 features collapsing to 6; Morris 16 features collapse to 6. The projection is lossy by design — we trade per-variable fidelity for a cross-testbed input shape that every detector can consume.

Discuss what's lost: HAI-specific sensor types (vibration, anonymous boiler codes) and Morris-specific comm metadata are excluded. Fraction of variance retained after projection: [to measure and report].

## 5.2 Finding 3.1 — source calibration collapses to the class-prior baseline

Under `src-cal` (threshold = 99th percentile of source validation scores), every transferred detector degenerates to a predict-all-attack baseline calibrated to the *target's* class prior.

| Target | Class prior (after windowing) | Predict-all F1 | Observed model F1 range |
|---|---|---|---|
| Morris | 0.91 | 0.95 | 0.95–0.99 (windowed SOTA); 0.62–0.75 (classical) |
| HAI    | 0.03 | 0.05 | 0.04–0.07 (all models) |

The score distributions on source and target data have no calibrated overlap; a fixed percentile threshold either flags everything or nothing. **A paper that reports only `src-cal` numbers on Morris (say F1=0.98) is reporting a class-prior artifact, not a transfer success.** This is — to our knowledge — the first explicit quantification of this failure mode in the ICS transfer literature.

Insert Figure F3 (`transfer_same_vs_cross.png`) and Figure F2's top row (`transfer_calibration_comparison.png`).

## 5.3 Finding 3.2 — target calibration reveals a classical-vs-SOTA split

Under `tgt-cal` (threshold from 99th percentile of *target-normal* validation — unlabeled, attack-free data available in any realistic deployment), the collapse disappears and we see a genuine separation:

| model | tgt-cal eTaPR | tgt-cal PA-F1 |
|---|---|---|
| Dense AE | 0.07 | 0.14 |
| OC-SVM | 0.08 | 0.17 |
| Isolation Forest | 0.13 | 0.22 |
| **LSTM-AE** | **0.44** | **0.81** |
| **TranAD** | **0.38** | **0.77** |
| **USAD** | **0.40** | **0.74** |

On HAI → Morris. The classical three are clustered near zero; the windowed SOTA three are clustered between 0.38 and 0.44 eTaPR. This is the first crisp classical-vs-SOTA gap in our study and it is **invisible under the more common source-calibrated evaluation.**

Reverse direction (Morris → HAI) is harder: pointwise F1 stays under 0.07 for all models because the HAI 2.4% attack prior saturates any 99th-percentile operating point. PA-F1 recovers for classical detectors (Dense AE 0.47, OC-SVM 0.36) via the segment-inflation mechanism critiqued in Chapter 4 — the same weakness that boosts classical PA-F1 same-dataset.

Insert Figure F2 (`transfer_calibration_comparison.png`).

### Discussion of why only the windowed models transfer

Hypothesis: the classical models' normal manifold is essentially per-dataset specific; they fit the density of the source type-vector without capturing the temporal structure that is approximately shared between testbeds. Windowed models (LSTM, USAD, TranAD) encode short-term dynamics of normal operation — pressure rising in ramp phases, valves opening after setpoints move — and those dynamics recur in both testbeds. The representation retains the invariant; the operating point does not.

## 5.4 Finding 3.3 — within-HAI LOPO isolates feature shift

Holding class prior fixed (both train and test are HAI), we drop one process's feature columns and compute F1 per-attack-target-process on the test split.

### 5.4.1 P2 drop produces specific blindness

| model | F1 on P1-attacks (drop-P2) | F1 on P2-attacks (drop-P2) |
|---|---|---|
| Dense AE | 0.41 | 0.03 |
| LSTM-AE | 0.09 | 0.00 |
| USAD | 0.28 | 0.07 |

P2 is the turbine controller; its sensors are almost all discrete state flags with no analogue in P1, P3, or P4. Removing them leaves the model blind to attacks that manifest only through those flags. The effect is **targeted** — detection on P1 attacks barely moves.

### 5.4.2 P1 drop produces global degradation

When P1 (water loop, 33 of 79 features, 82% of test attacks) is removed, all three models drop to F1 ~ 0.1 on every attack class. P1 carries most of the model's picture of "normal"; losing those features starves the whole detector rather than producing a specific blindness.

### 5.4.3 P3 / P4 drop is mild and sometimes helpful

Dropping P3 or P4 occasionally *improves* detection: USAD drop-P3 F1 on P1-attacks = 0.57 (vs. 0.35 in the Phase-2 same-domain baseline). Narrower input regularizes the windowed models, a small second-order finding.

### 5.4.4 No consistent SOTA advantage under LOPO

Unlike the cross-testbed setting (where SOTA clearly beat classical under tgt-cal), LOPO shows no consistent classical/SOTA gap. Dense AE is competitive with USAD on the P2-blindness diagnostic; both beat LSTM-AE on absolute numbers under drop-P2. The cross-testbed SOTA advantage therefore looks tied to *cross-schema representation*, not to robustness to feature loss per se.

Insert Figure F4 (`lopo_heatmap.png`).

## 5.5 Cross-reading: sensor-level transfer ceilings

Best within-HAI LOPO pointwise F1 = 0.57 (USAD, drop-P3, P1-attacks). Best cross-testbed `tgt-cal` pointwise F1 = ~0.13 (LSTM-AE HAI→Morris). Same architecture, same class-prior control, partial sensor overlap: the cross-dataset ceiling is an order of magnitude lower than the within-dataset one. This defends the cross-testbed numbers against the "you just need a better common feature space" reviewer critique — even under the best within-testbed conditions, sensor-level transfer loses half its detection capability.
