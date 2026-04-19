# Thesis Outline

**Working title:** *Cross-Testbed, Time-Aware, Attributable Anomaly Detection in Industrial Control Systems*

Target: undergraduate diploma thesis; secondary target workshop paper (DSN-W / ESORICS CPS-SPC / NDSS workshop). The three contributions drive three distinct results sub-chapters, each headlined by one of the findings below.

---

## Abstract (~200 words)

Current ICS anomaly detectors report high F1 on the dataset they were trained on, but (a) their rankings change under time-aware evaluation metrics, and (b) they fail to generalize across testbeds — both effects obscured by widely-used headline numbers. This work benchmarks six detectors (Isolation Forest, One-Class SVM, Dense AE, LSTM-AE, USAD, TranAD) on HAI 21.03 and the Morris gas-pipeline dataset under three time-aware metrics, runs a cross-testbed transfer study isolating representation transfer from threshold transfer, and evaluates per-sensor attribution at the process level.

**Three contributions:**
- A cross-testbed transfer protocol that separates representation transfer from threshold transfer; windowed SOTA models achieve ~0.4 eTaPR on HAI→Morris under target calibration but collapse to the predict-all baseline under source calibration.
- A time-aware metric comparison showing PA-F1 inflates small-area detections while eTaPR tracks event-level recall, and a within-HAI leave-one-process-out experiment isolating feature-distribution shift.
- A detection-vs-localization tradeoff: Dense AE is the weakest detector but the only model whose per-feature attribution beats a random baseline on all three attacked HAI processes.

---

## Chapter map

| # | Chapter | Primary source material | Target length |
|---|---|---|---|
| 1 | Introduction | §2 thesis statement, three contributions | 3–4 pp |
| 2 | Background | Survey: Shin 2020 (HAI), Tuli 2022 (TranAD), Audibert 2020 (USAD), Morris 2015; metrics Xu 2018 / Kim 2022 / Hwang 2022 | 4–6 pp |
| 3 | Method | `src/` architecture: ABC, loaders, preprocessing, schema-align, LOPO, attribution eval | 6–8 pp |
| 4 | Results I — Detection & metrics | Phase 1–2 grids; notebook 04, 06 | 4–5 pp |
| 5 | Results II — Cross-testbed transfer & LOPO | Phase 3; notebook 05 | 6–8 pp (headline chapter) |
| 6 | Results III — Attribution | Phase 4; notebook 07 | 4–5 pp |
| 7 | Discussion & limitations | All findings combined, reviewer preempts | 3–4 pp |
| 8 | Conclusion | Recap + future work | 1–2 pp |

Total ~30–40 pp before references and appendix.

---

## Writing order (draft → revise → polish)

Weeks 13–14 (draft):
1. Chapter 3 (Method) first — it's the most deterministic: describe what's in `src/`.
2. Chapter 4 (Results I) next — Phase 1–2 numbers are stable.
3. Chapter 5 (Results II) — biggest chapter; map each Phase-3 finding to a subsection.
4. Chapter 6 (Results III) — Phase-4 attribution.
5. Chapter 2 (Background) — written with full mental model of results.
6. Chapter 1 (Introduction) & Chapter 7 (Discussion) — written last, citing the specific numbers now in hand.
7. Abstract last (convention; it's a forward-pointer from finished text).

Weeks 15–16:
- One full revision pass for voice/consistency.
- Figure polish (axis labels, caption self-contained).
- Bibliography tightening.
- Defence slides + rehearsal.

---

## Figures to produce or polish

All scripted; paths below relative to `results/figures/`.

| ID | Source notebook | Path | Chapter |
|---|---|---|---|
| F1 | 04 | `04_sota/sota_six_model_comparison.png` | 4 |
| F2 | 05 | `05_cross_dataset/transfer_calibration_comparison.png` | 5 |
| F3 | 05 | `05_cross_dataset/transfer_same_vs_cross.png` | 5 |
| F4 | 05 | `05_cross_dataset/lopo_heatmap.png` | 5 |
| F5 | 07 | `07_attribution/attribution_p_at_5.png` | 6 |

To add (nice-to-have):
- Per-metric ranking-change plot (Chapter 4) — show how eTaPR vs. PA-F1 swaps model ranks.
- Attention-rollout vs. reconstruction scatter (Chapter 6) to visualize Finding 4.2.

---

## Tables

| T | Content | Chapter |
|---|---|---|
| T1 | 6 × 2 × 3 detection grid (models × datasets × metrics) | 4 |
| T2 | src-cal vs. tgt-cal vs. target-same baseline | 5 |
| T3 | LOPO per-attack-process F1 | 5 |
| T4 | Attribution precision@k and lift over random | 6 |

---

## Risks / known gaps

- TranAD numbers are ~0.5 PA-F1 below Tuli 2022's. Documented, not hidden; Chapter 4 will call this out and the discussion will argue why our qualitative finding (TranAD #1 on HAI PA-F1 in our setup) is still informative.
- LSTM-AE and USAD give near-identical attribution. Called out as a structural coincidence; without multi-seed runs it can't be fully ruled out as a seed artifact.
- Attribution is process-level, not per-sensor. HAI 21.03 doesn't publish machine-readable per-attack sensor-target lists. Discussion flags a HAI 22.04 follow-up.
- Single seed per model for attribution; core detection has multi-seed for classical models only.
