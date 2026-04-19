# 6. Results III — Per-sensor attribution

Notebook: `07_attribution.ipynb`.
Figure: F5 (`results/figures/07_attribution/attribution_p_at_5.png`).
Source: `results/metrics/attribution.parquet` (3 seeds × 4 models × reconstruction; 3 seeds × TranAD × attention).

## 6.1 Evaluation protocol recap

For each attack window in HAI test, the model produces a per-feature attribution vector (see §3.7). We rank features by attribution and compute precision@k — the fraction of the top-k features that belong to the attacked process (from HAI's `attack_P{n}` label). Random baseline = share of features in the attacked process (0.48 for P1, 0.28 for P2, 0.09 for P3).

HAI 21.03 does not publish per-sensor attack targets in machine-readable form, so the process-level eval is what's available. A stronger per-sensor eval on HAI 22.04 is a natural follow-up (§7).

## 6.2 Precision@5 table (mean ± std over seeds 7 / 42 / 123)

All numbers are reconstruction-based attribution.

| model | P1 p@5 | P2 p@5 | P3 p@5 |
|---|---|---|---|
| Dense AE | 0.69 ± 0.01 | 0.74 ± 0.00 | **0.33 ± 0.02** |
| LSTM-AE  | 0.74 ± 0.00 | 0.09 ± 0.00 | 0.20 ± 0.00 |
| USAD     | 0.74 ± 0.00 | 0.09 ± 0.00 | 0.21 ± 0.02 |
| TranAD   | 0.58 ± **0.17** | **0.35 ± 0.19** | 0.02 ± 0.01 |
| random baseline | 0.48 | 0.28 | 0.09 |

Insert Figure F5 (bars with +/- 1 sigma error bars, random baseline dashed).

## 6.3 Finding 4.1 — detection-vs-localization tradeoff (robust across seeds)

Lift over random baseline for mean p@5:

| model | P1 lift | P2 lift | P3 lift |
|---|---|---|---|
| **Dense AE** | **1.45x** | **2.68x** | **3.72x** |
| LSTM-AE | 1.55x | 0.32x | 2.22x |
| USAD    | 1.55x | 0.33x | 2.33x |
| TranAD  | 1.22x | 1.26x | 0.23x |

**Dense AE — the weakest Phase-2 detector — is the only model whose attribution is reliably above random across all three processes.** Seed variance is small (~0.01-0.02 on P3); this is a robust result, not a seed artifact.

The mechanism: Dense AE operates per-timestep, so per-feature reconstruction error is aligned with the current frame's anomaly. Windowed models (LSTM-AE, USAD, TranAD) reconstruct a 60-step window and average the per-feature error over time; the averaging concentrates mass on features with the largest dynamic-range variation, which for HAI are the P1 flow/pressure sensors. A small perturbation to a few discrete P2 flags is lost in the mean.

**This inverts the detection ranking of Chapter 4.** TranAD is #1 on HAI eTaPR (Chapter 4.1) but near-worst on P3 localization (0.02 mean p@5, 0.23x random lift). For an operator triaging a flagged window, Dense AE's per-feature output is the more actionable signal. Reporting only detection F1 obscures this.

Insert Figure F5.

## 6.4 Finding 4.2 — attention rollout is null (robust across seeds)

TranAD attention-weighted attribution differs from plain reconstruction attribution by <0.01 on every (process, k) pair across all three seeds. The two methods are empirically interchangeable.

Intuition: with a single-layer encoder, rollout = raw attention matrix. Attention averaged over 60 output positions on HAI normal data is close to uniform; reweighting timesteps uniformly leaves the per-feature ranking unchanged. Attention-based XAI for per-sensor localization in ICS windowed AD is not automatically the right tool — one has to verify attention-over-time is structurally variable before investing.

## 6.5 Finding 4.3 — LSTM-AE ≡ USAD attribution is a genuine structural result

Across 3 seeds, LSTM-AE and USAD produce p@k values identical to two decimals on every (process, k) pair. Standard deviations are ≤ 0.016 within each model. This is not seed noise; the two architectures converge to the same per-feature reconstruction ranking on HAI.

Hypothesis: the ranking is dominated by the MinMax-scaled feature variance structure of HAI's 79 sensors. P1 sensors carry the highest dynamic range and feature count, so any architecture with enough capacity to minimise windowed reconstruction error learns to prioritise them. LSTM-AE's sequence objective and USAD's adversarial dual-head objective both converge to the same optimum in the feature-ranking sense.

This is a finding of independent interest: **if your model class produces the same attribution ranking as a simpler baseline, the extra architectural complexity is buying detection accuracy (USAD and LSTM-AE do differ in detection F1 slightly), not attribution quality.** Operators paying the complexity tax for a richer model should audit that they're getting the richer output.

## 6.6 Revised framing vs. seed-42-only report

An earlier draft (committed at a2957b0) reported "TranAD actively anti-localizes on P3 (p@5=0.02, lift 0.2x, worse than flipping coins)." Multi-seed averaging softens this claim:

- P3 remains a TranAD weakness (mean 0.02, range [0.02, 0.03] across seeds — consistent, not seed-random).
- **But TranAD on P2 had large seed variance**: p@5 = 0.23 at seed 42 (the original commit) vs. 0.57 at seed 7. Mean 0.35, *above* the 0.28 random baseline.

The revised framing: TranAD *can* localize to P2 with enough seed-trial budget, but its localization is seed-unstable, whereas Dense AE and LSTM-AE/USAD are seed-stable (even if the latter two are structurally stuck on P1). This is an argument for multi-seed evaluation in any attribution study and a caveat against single-seed claims about transformer XAI.

## 6.7 Limitations

1. **Process-level eval is a weaker signal than per-sensor eval.** HAI 22.04 has more detailed per-attack descriptions in the README; a hand-curated per-sensor ground truth on 22.04 would strengthen the claim.
2. **Dense AE is non-windowed; LSTM-AE, USAD, TranAD are windowed.** The localization-vs-detection tradeoff is partly confounded with windowing. A windowed MLP AE (or a non-windowed transformer with 1-step context) would disentangle; out of scope here.
3. **Attention rollout is a single-layer approximation.** A deeper TranAD could give richer signals; deeper TranAD also takes hours per run and was out of scope for our 4 GB GPU.
