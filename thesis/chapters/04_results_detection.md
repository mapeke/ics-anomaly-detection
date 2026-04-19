# 4. Results I — Same-dataset detection and metric sensitivity

Notebooks: `04_sota_results.ipynb`, `06_metric_sensitivity.ipynb`.
Source of numbers: `results/metrics/summary.parquet`, same-dataset rows.

## 4.1 Full 6 × 2 × 3 grid

| Model | HAI eTaPR | HAI PA-F1 | HAI pointwise | Morris eTaPR | Morris PA-F1 | Morris pointwise |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.243 | 0.621 | 0.185 | 0.145 | 0.297 | 0.090 |
| One-Class SVM    | 0.170 | 0.540 | 0.265 | 0.369 | 0.631 | 0.275 |
| Dense AE         | 0.057 | 0.477 | 0.356 | 0.356 | 0.586 | 0.253 |
| LSTM-AE          | 0.147 | 0.221 | 0.105 | — (degenerate) | — | — |
| USAD             | 0.151 | 0.222 | 0.107 | 0.009 | 0.030 | 0.001 |
| TranAD           | **0.276** | 0.458 | 0.189 | 0.339 | 0.647 | 0.129 |

Insert Figure F1 (`results/figures/04_sota/sota_six_model_comparison.png`).

## 4.2 Ranking changes across metrics

The rankings are not stable:
- On HAI: best eTaPR = TranAD; best PA-F1 = Isolation Forest (0.621); best pointwise = Dense AE.
- On Morris: best eTaPR = OC-SVM (0.369); best PA-F1 = TranAD (0.647); best pointwise = OC-SVM.

Using only one metric, a reader could rank any of OC-SVM, TranAD, Isolation Forest, or Dense AE as the "winner" — and publish a confident title. **This is why we report three.**

Insert figure: ranking-change visualization (Kendall-tau-style). To add to notebook 06.

## 4.3 PA-F1 inflation on classical detectors

Dense AE HAI: pointwise 0.356, PA-F1 0.477 (1.34× inflation).
Isolation Forest HAI: pointwise 0.185, PA-F1 0.621 (3.35× inflation).
OC-SVM Morris: pointwise 0.275, PA-F1 0.631 (2.29× inflation).

Classical models produce isolated high-score points rather than contiguous positive runs; PA-F1 grants full segment credit to those isolated flags, inflating the headline numbers disproportionately. eTaPR penalises segment coverage and reduces the inflation. This is the Kim-2022 critique replicated on our data.

## 4.4 USAD collapse on Morris

USAD eTaPR on Morris = 0.009. The model's test-time scores are effectively constant — the discrete Modbus features break the USAD assumption of a continuous normal manifold. Audibert et al. 2020 did not evaluate on a Modbus RTU dataset; we flag this as a reviewer-facing pitfall for researchers extending USAD to protocol-level ICS data.

## 4.5 HAI LSTM-AE reproducibility check

Seed 42, val-percentile threshold 99, 10 epochs on a 30k window subsample: pointwise 0.105, PA 0.221, eTaPR 0.147. Shin et al. 2020 report ~0.37–0.45 eTaPR for LSTM baselines on HAI 20.07. The gap is ~0.2 eTaPR; it is outside the ±0.05 CLAUDE.md tolerance. Sources of the gap:
1. Dataset version 20.07 vs 21.03 (different attack set).
2. 30k subsample of ~1.5M windows for CPU tractability.
3. Conservative threshold (99th percentile vs. oracle best-F1 the paper uses).

Per project policy, the gap is documented rather than tuned. The qualitative ordering (windowed deep models > classical on HAI eTaPR) is preserved.

## 4.6 HAI TranAD reproducibility check

Seed 42, RTX 3050 Ti GPU, 30 epochs, 30k window subsample: pointwise 0.189, PA 0.458, eTaPR 0.276. Tuli et al. 2022 report PA-F1 ~0.97 on HAI 20.07. The gap is ~0.5 PA-F1 — large. Sources are the same as §4.5 plus:
4. Single-layer encoder with d_model=64 (our config) vs. paper's deeper/wider network.
5. Oracle-PA-best threshold in the paper.

The qualitative ranking still holds: TranAD is #1 on HAI under PA-F1 in our setup. We argue the value of the result is in the comparison across models under a consistent protocol, not in matching published absolutes.
