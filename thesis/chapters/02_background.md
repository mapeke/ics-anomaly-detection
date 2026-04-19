# 2. Background

## 2.1 ICS threats and datasets

Cover:
- ICS threat model — integrity vs. availability; why "anomaly" maps to attack on physical plant.
- Public datasets comparison: HAI (iksdataset.org releases 20.07 → 23.05), SWaT, WADI, Morris.
- Why we choose HAI 21.03 + Morris: complementary (continuous sensors vs. discrete Modbus frames), different plant topologies, different attack mixes.
- Size, attack-rate, and time-resolution table. Cite:
    - Shin et al. 2020, "HAI Security Dataset," USENIX CSET.
    - Morris & Gao 2014, "Industrial Control System Traffic Data Sets for Intrusion Detection Research," ICCIP.

## 2.2 Detectors used in this thesis

One short paragraph per detector. For each, cite the original paper, the training objective, and the reason it's a reasonable baseline/SOTA choice.

- **Isolation Forest** (Liu et al. 2008) — tree-based density proxy; strong classical baseline.
- **One-Class SVM** (Schölkopf et al. 2001) — kernel density baseline; well-established in ICS literature.
- **Dense Autoencoder** — tabular per-timestep reconstruction; simplest deep baseline.
- **LSTM Autoencoder** — windowed sequence reconstruction; captures short-term dependencies.
- **USAD** (Audibert et al. 2020, KDD) — adversarially-trained dual autoencoder over sliding windows.
- **TranAD** (Tuli et al. 2022, VLDB) — transformer encoder with two-phase adversarial decoder and attention-based attribution.

Acknowledge reproducibility gap on TranAD (§4 reports the number; §7 discusses).

## 2.3 Time-aware evaluation metrics

- **Point-wise P/R/F1** — standard sklearn; no concept of events.
- **Point-adjust F1** (Xu et al. 2018, KDD) — any single point inside an attack event grants full segment credit. The AAAI 2022 critique (Kim et al.) demonstrates the inflation formally: a random predictor with 1% positive rate achieves PA-F1 > 0.5 on typical ICS benchmarks.
- **eTaPR** (Hwang et al. 2022) — event-weighted ePrecision / eRecall / eF1 that grant partial segment credit weighted by overlap and duration. Reduces the PA-F1 inflation while still rewarding "flag at least part of the event."

Include the Kim-2022 PA-F1 inflation example as a grounding illustration. Note our policy (CLAUDE.md §5.3): every number reported under all three metrics; ranking changes across metrics treated as a finding, not noise.

## 2.4 Cross-dataset / transfer learning in time-series AD

Brief survey of what's been tried:
- Same-domain fine-tuning (cited e.g. Deng & Hooi 2021 for GDN).
- Autoencoder-based domain adaptation.
- Schema-align for tabular transfer.

Position this thesis: we do the simplest defensible scheme (per-variable type tagging, CLAUDE.md §5.2 option 1) and ask whether it transfers. Richer schemes (learned projections, adversarial alignment) remain on the shelf.

## 2.5 Attribution / per-sensor localization

- Reconstruction-error per-feature attribution for AE family — trivial, standard.
- Attention rollout (Abnar & Zuidema 2020) for transformers — used in NLP; we adapt to ICS and report a null finding (Chapter 6).
- SHAP / integrated gradients are alternative approaches (cite briefly, note they're compute-heavy for long windows).

Process-level evaluation: precision@k of top-k attributed features landing on the attacked process. Weaker than per-sensor ground truth (which HAI 21.03 does not publish), but uses the labels we have.
