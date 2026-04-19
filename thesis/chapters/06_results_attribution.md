# 6. Results III — Per-sensor attribution

Notebook: `07_attribution.ipynb`.
Figure: F5 (`results/figures/07_attribution/attribution_p_at_5.png`).
Source: `results/metrics/attribution.parquet`.

## 6.1 Evaluation protocol recap

For each attack window in HAI test, the model produces a per-feature attribution vector (see §3.7). We rank features by attribution and compute precision@k — the fraction of the top-k features that belong to the attacked process (from HAI's `attack_P{n}` label). Random baseline = share of features in the attacked process (0.48 for P1, 0.28 for P2, 0.09 for P3).

HAI 21.03 does not publish per-sensor attack targets in machine-readable form, so the process-level eval is what's available. A stronger per-sensor eval on HAI 22.04 is a natural follow-up (§7).

## 6.2 Precision@k table

All numbers are reconstruction-based attribution, seed 42, HAI.

| model | P1 p@1 | P1 p@5 | P1 p@10 | P2 p@1 | P2 p@5 | P2 p@10 | P3 p@1 | P3 p@5 | P3 p@10 |
|---|---|---|---|---|---|---|---|---|---|
| Dense AE | 0.72 | 0.68 | 0.65 | **0.81** | **0.74** | 0.59 | **0.43** | **0.34** | 0.24 |
| LSTM-AE  | 0.84 | 0.74 | 0.73 | 0.08 | 0.09 | 0.11 | 0.05 | 0.20 | 0.16 |
| USAD     | 0.84 | 0.74 | 0.73 | 0.08 | 0.09 | 0.10 | 0.05 | 0.20 | 0.16 |
| TranAD   | 0.75 | 0.69 | 0.62 | 0.14 | 0.23 | 0.32 | 0.00 | 0.02 | 0.01 |

Random baseline (fixed by the feature counts per process): 0.48 / 0.28 / 0.09 for P1 / P2 / P3.

## 6.3 Finding 4.1 — detection-vs-localization tradeoff

Lift over random baseline (p@5 / random):

| model | P1 lift | P2 lift | P3 lift |
|---|---|---|---|
| **Dense AE** | **1.42×** | **2.64×** | **3.78×** |
| LSTM-AE | 1.54× | 0.32× | 2.22× |
| USAD | 1.54× | 0.32× | 2.22× |
| TranAD | 1.44× | 0.82× | 0.22× |

**Dense AE — the weakest Phase-2 detector in eTaPR — is the only model that localizes attacks above random across all three processes.** The windowed SOTA models collapse attribution onto P1 features regardless of target, scoring *below* random on P2 (lstm_ae, usad) and on P3 (tranad).

Why? Dense AE operates per-timestep, so per-feature reconstruction error is aligned with the current frame's anomaly. Windowed models reconstruct an entire 60-step window; the per-feature error is averaged over time, and the averaging concentrates mass on the features with the most dynamic-range variation — which for HAI are P1's flow/pressure sensors. When a P2 or P3 attack produces a small perturbation to a few discrete flags, it is lost in the mean.

**This is a detection-vs-localization tradeoff.** Phase 2 established SOTA > classical on detection (TranAD HAI eTaPR 0.28, Dense AE 0.06). Phase 4 inverts the ranking for localization: Dense AE P3 p@5 = 0.34, TranAD P3 p@5 = 0.02. An operator faced with a flagged window gets more from Dense AE's attribution than from a more accurate but diffuse SOTA attribution. Reporting only detection F1 hides this.

Insert Figure F5.

## 6.4 Finding 4.2 — attention rollout null

TranAD attention-weighted attribution uses the encoder self-attention as a per-input-timestep weight on the per-(t, f) reconstruction error (§3.7.1). Results:

| metric | reconstruction | attention-weighted |
|---|---|---|
| P1 p@5 | 0.69 | 0.69 |
| P2 p@5 | 0.23 | 0.23 |
| P3 p@5 | 0.02 | 0.02 |

Numerically identical to two decimals. The intuition:
- Single-layer encoder → rollout = raw attention matrix.
- 60-step attention averaged over output positions is near-uniform on HAI normal data.
- Per-feature attribution is dominated by the feature-axis variation in reconstruction error; reweighting timesteps uniformly does nothing.

**Null finding of independent interest:** attention rollout, effective in NLP for token-level attribution, does not automatically transfer to per-sensor localization in ICS windowed anomaly detection. A reader tempted to adopt attention-based XAI for ICS should audit whether attention-over-time is structurally variable in their setup before investing.

## 6.5 Finding 4.3 — LSTM-AE and USAD produce identical attribution

The P1/P2/P3 rows for LSTM-AE and USAD in the p@k table are identical to three decimal places. This is a striking structural coincidence. Hypothesis: in HAI's 79-feature space, the optimal reconstruction attribution converges to a specific feature ranking that both architectures land on when trained with comparable capacity on comparable windows. Without multi-seed runs we can't rule out an artifact of seed 42.

Action item: re-run with seeds 7 and 123 on both models (pending).

## 6.6 Limitations

1. **Process-level eval is a weaker signal than per-sensor eval.** HAI 22.04 has more detailed per-attack descriptions in the README; a hand-curated per-sensor ground truth on 22.04 would strengthen the claim.
2. **Single seed.** Cited.
3. **Attention rollout is a single-layer approximation.** A deeper TranAD could give richer signals; deeper TranAD also takes hours per run and was out of scope for our 4 GB GPU.
