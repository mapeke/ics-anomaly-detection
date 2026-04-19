# 2. Background

## 2.1 Industrial control systems and public datasets

Industrial control systems (ICS) mediate between digital policy and physical plant — programmable logic controllers (PLCs), distributed control systems (DCSs), remote telemetry units (RTUs), and the Modbus/OPC-UA/DNP3 protocols that link them. An intrusion in this regime differs from a conventional IT compromise in that the consequence is physical: opened valves, mis-commanded pumps, transformer failure. Stuxnet (Iran, 2010), the Maroochy Water breach (Australia, 2000), and the two Ukraine grid incidents (2015, 2016) are the commonly-cited precedents for why ICS security is a first-class research area [@langner2013stuxnet; @ukraine2016analysis].

Public ICS datasets supply the benchmarks under which anomaly-detection work is evaluated. The four most commonly used in the recent deep-learning literature are:

| Dataset | Testbed | Features | Attack labelling | Typical use |
|---|---|---|---|---|
| HAI 20.07 / 21.03 / 22.04 / 23.05 | HIL-based boiler + turbine + water + simulation | 59–86 sensors @ 1 Hz | per-process flags (`attack_P1`..`P4`) | most popular in 2022–2026 deep-AD papers |
| SWaT (Goh et al. 2017) | Water treatment plant | 51 features @ 1 Hz | binary label, no target decomposition | heavily used pre-2022 |
| WADI (Ahmed et al. 2017) | Water distribution network | 123 features @ 1 Hz | binary label | companion to SWaT |
| Morris gas pipeline (Morris & Gao 2014) | Laboratory gas-pipeline testbed over Modbus RTU | 16 features / frame | "binary result" + "specific result" (attack type) | classical ICS AD and IDS work |

We evaluate on HAI 21.03 and Morris. Rationale:

1. **Complementary physical regimes.** HAI is continuous-sensor data from a multi-process HIL rig; Morris is discrete-frame Modbus RTU from a single-process gas-pipeline testbed. A detector that transfers between them is genuinely "ICS-general"; one that doesn't is testbed-specific.
2. **Complementary attack labelling.** HAI's per-process `attack_P{n}` tagging is the only public ICS dataset with machine-readable per-sub-system attack decomposition; it is what makes the LOPO and attribution studies in Chapters 5 and 6 possible. Morris's `specific result` column gives attack *type* (denial-of-service, command injection, reconnaissance) but not sensor-target.
3. **Different class priors.** HAI test has a 2.4% attack rate; Morris has ~48% (91% after 60-step windowing). The pair stresses the threshold-calibration question that Chapter 5 is built around.
4. **Availability.** HAI 21.03 is cloneable from the official repository without Git LFS; HAI 22.04+ require LFS and we were not guaranteed quota. Morris is a single ARFF file. CLAUDE.md §5.1 documents the choice.

SWaT and WADI are excluded not because they lack value — they are the standard benchmarks for SOTA detector papers — but because adding more single-testbed datasets does not change Contribution 1 (cross-testbed generalization). An extension to SWaT-HAI transfer is future work; we argue in §7.5 that the cross-testbed claim already rests on HAI-Morris.

## 2.2 Anomaly detectors evaluated

Six detectors span the classical-to-SOTA range. For each we fit the PyTorch-or-sklearn implementation on normal-only training data, score test windows, and threshold on target-val percentile (Chapter 3).

**Isolation Forest** [@liu2008isolation]. Random-forest density proxy: anomalies reach leaf nodes faster than normal points. Strong, near-parameter-free baseline for tabular data; we include it under the same-dataset detection grid and the cross-testbed schema-align projection.

**One-Class SVM** [@scholkopf2001estimating]. Kernel density baseline from the pre-deep era but still commonly cited in ICS AD. The RBF kernel's distance-sensitivity makes it a counterpoint to Isolation Forest's partitioning view.

**Dense autoencoder**. The simplest deep baseline: a fully-connected autoencoder trained to reconstruct per-timestep feature vectors. Anomaly score = per-sample MSE. Per-feature reconstruction MSE is the attribution path and ends up being the most useful localization signal in the study (Chapter 6).

**LSTM autoencoder** [@malhotra2016lstm]. Sequence-reconstruction baseline using an LSTM encoder-decoder over 60-step windows. Widely reported on HAI in the early ICS-AD literature.

**USAD** [@audibert2020usad], KDD 2020. Two MLP-based autoencoders share an encoder and are trained adversarially: decoder 1 minimises reconstruction, decoder 2 maximises it, inference scores combine both. The adversarial head is the paper's contribution over a plain windowed AE.

**TranAD** [@tuli2022tranad], VLDB 2022. Transformer encoder shared between two decoders; Phase 2 of the two-phase scheme passes a per-element reconstruction residual back to the encoder as a focus signal. The paper's reported HAI 20.07 numbers are the current SOTA cite-target in the literature. Our implementation diverges from the reference in documented ways (see `src/models/tranad.py` header and §4.6); our HAI 21.03 numbers are ~0.5 PA-F1 below the paper's 20.07 numbers, which we discuss honestly rather than tuning to fit.

The three classical detectors (IF, OC-SVM, Dense AE) operate per-timestep; LSTM-AE, USAD, TranAD operate on 60-step sliding windows. The windowing difference matters for Chapter 5 (windowed models are the ones that transfer under target calibration) and Chapter 6 (windowed models fail to localize).

## 2.3 Time-aware evaluation metrics

The three metrics in this study are standard in time-series anomaly detection and disagree systematically.

**Point-wise precision / recall / F1.** sklearn default. Treats every timestep as independent: a flag at time t is correct iff y_t = 1. This under-reports detection quality when an attack consists of many consecutive abnormal points and the detector flags only a subset, and over-reports it when the detector is lucky with isolated hits.

**Point-adjust F1** (PA-F1) [@xu2018unsupervised]. Addresses the "many consecutive points" problem by granting credit for the whole attack segment if any single point inside it is flagged:
```
if any y_hat_t == 1 for t in [event_start, event_end]:
    set y_hat_t = 1 for all t in that range
compute F1 on the adjusted y_hat vs. y
```
Used in most recent deep ICS AD papers including TranAD. Kim et al. [@kim2022towards] (AAAI 2022) show that PA-F1 is inflated under reasonable conditions: a random predictor with 1% positive rate achieves PA-F1 > 0.5 on typical event distributions. The paper frames this as a methodological critique rather than a proposal for a replacement.

**eTaPR** [@hwang2022etapr]. Event-weighted precision and recall with a partial-credit rule: a flagged segment that overlaps an attack event gets credit proportional to the overlap's fraction of the event, and a flagged run that spans multiple events splits its credit. Reduces PA-F1 inflation while still rewarding "flag at least some of the event." We port the reference implementation and verify against the original authors' published numbers on a known model as a sanity check (see `src/evaluation/etapr.py`).

The thesis policy (CLAUDE.md §5.3): every number is reported under all three metrics. Ranking changes across metrics are findings (Chapter 4.2), not noise.

## 2.4 Cross-dataset transfer in time-series anomaly detection

Most published ICS AD papers evaluate on a single testbed. The narrower "transfer within a testbed but across time" setting — train on one HAI split, test on another — is common; cross-testbed transfer is rare.

The obvious obstacle is *schema mismatch*: HAI's 79 sensors have no overlap with Morris's 16 Modbus fields at the raw-name level. The literature offers three broad approaches:

1. **Per-variable type tagging** — hand-map each raw feature to a canonical type (pressure, flow, valve, etc.) and aggregate same-typed features into a shared vector. Simplest; relies on domain knowledge for tagging quality. Used in Kravchik & Shabtai 2018 informally and formalised elsewhere for SCADA benchmarks.
2. **Learned projection** — train an encoder (contrastive or autoencoder-style) to project both datasets into a shared latent space using only normal windows. Flexible; requires a second training phase and adds a methodological defense burden.
3. **Punt and do within-testbed LOPO** — leave one process out within the source testbed. Weakest, but clean.

We take option 1 as the primary study (CLAUDE.md §5.2) and option 3 as a within-testbed control. Option 2 remains listed in §7.5 as follow-up. Our contribution over prior option-1 work is the **calibration-regime split** (source-cal vs. target-cal, Chapter 5), which isolates representation transfer from threshold transfer — an axis that, to our reading, the published schema-align transfer literature has not evaluated explicitly.

## 2.5 Attribution and per-sensor localization

Attribution for anomaly detection answers *which sensor* the detector is reacting to at a flagged timestep.

**Per-feature reconstruction error.** For autoencoder-family models, the per-feature squared reconstruction error is a natural attribution: the feature whose value deviates most from the reconstruction contributes most to the flag. Widely used informally; we evaluate it formally in Chapter 6.

**Attention rollout** [@abnar2020quantifying]. For transformer-based detectors, attention weights at each layer are interpreted as "how much the output attends to the input," and rollout aggregates across layers as an identity-plus-averaged matrix product. In NLP the method is well-established for token-level attribution; its transfer to time-series AD has been less tested. We adapt rollout to a single-layer TranAD encoder (§3.7.1) and report a null finding (Chapter 6.4).

**Gradient-based methods.** SHAP [@lundberg2017unified], integrated gradients [@sundararajan2017axiomatic], and saliency maps are alternatives but compute-heavy for 60-step windows on multi-output models; we note them and defer.

**Evaluation.** HAI publishes per-attack tags at the *process* level (`attack_P1`..`P4`) but not per-sensor target lists in machine-readable form. We therefore evaluate at the process level: fraction of top-k attributed features whose column name is prefixed by the attacked process. Random baseline equals the share of features in that process. A sensor-level eval would require parsing the HAI 22.04 README's qualitative per-attack descriptions into a hand-curated ground truth; this is listed as future work (§7.5). Process-level is weaker but uses the labels we have, and the lift-over-random framing keeps the claim well-calibrated.
