# 1. Introduction

## 1.1 Motivation

Industrial control systems (ICS) are the interface between digital policy and physical plant; an undetected intrusion can cause damage that software-only compromises cannot (Stuxnet, Maroochy, Ukraine 2015). Public ICS datasets — HAI (Shin et al. 2020/2021), SWaT (Goh et al. 2017), WADI (Ahmed et al. 2017), Morris (Morris & Gao 2015) — enable reproducible benchmarking, and a large literature of deep-learning anomaly detectors reports high F1 scores on them. This thesis takes three questions seriously that the headline numbers skip over:

1. **Do the rankings survive time-aware evaluation?** Point-wise F1 treats every timestep as independent. Point-adjust F1 (Xu et al. 2018) inflates scores by granting full credit when any single point inside an attack event is flagged. eTaPR (Hwang et al. 2022) attempts to restore event-level rigour. Which metric one reports changes which model "wins."
2. **Do these detectors generalize across testbeds?** All cited papers train and test on the same dataset. An HAI-trained model should, in principle, capture ICS-general structure and transfer to an industrial gas-pipeline testbed — or the claim of "anomaly detection" is overstated.
3. **Can the detector localize?** A flagged window is actionable only if an operator can see *which sensor* is implicated. We evaluate per-feature attribution against HAI's per-process attack labels.

## 1.2 Contributions

**C1 — Cross-testbed transfer protocol that separates representation from threshold.** We project HAI (79 features) and Morris (16 features) onto a shared 6-dim type-vector (pressure, pump_state, setpoint, system_state, valve_position, control_signal) and run transfer under two calibration regimes: source-domain percentile (`src-cal`) and target-normal percentile (`tgt-cal`). We show that `src-cal` collapses every detector to the predict-all trivial baseline of the *target* class prior — a known artifact but, to our knowledge, unquantified in the ICS transfer literature. Under `tgt-cal`, windowed SOTA models (LSTM-AE, USAD, TranAD) achieve 0.38–0.44 eTaPR on HAI→Morris, the first clear classical-vs-SOTA separation in our study.

**C2 — Within-HAI leave-one-process-out (LOPO) experiment isolating feature shift.** Holding the class prior fixed, we drop one process's features (P1, P2, P3, or P4) and measure per-attacked-process F1. Dropping P2 (turbine controller, discrete state flags) causes targeted blindness to P2-targeting attacks (F1 drops from 0.41 on P1-attacks to 0.03 on P2-attacks for Dense AE); dropping P1 (water loop, 82% of test attacks) causes global degradation. This sets a within-testbed ceiling on what sensor-level transfer can achieve, and contextualizes the lower cross-testbed numbers.

**C3 — Process-level attribution evaluation with a detection-vs-localization tradeoff.** Every model exposes `attribute(X) → (N, F)`; we evaluate precision@k of the top-k attributed features against HAI's per-process attack labels. Dense AE — the weakest Phase-2 detector — is the only model whose attribution beats random baseline on all three attacked processes (lift 1.4×, 2.6×, 3.8× at p@5). The windowed SOTA models concentrate attribution on P1 features regardless of target, and attention-weighted rollout for TranAD produces numerically identical numbers to plain reconstruction attribution — a null finding that questions the usefulness of attention-XAI for per-sensor ICS localization.

## 1.3 Structure

Chapter 2 surveys datasets, detectors, and metrics used in the study. Chapter 3 describes the implementation: unified `AnomalyDetector` interface, scaler-leak guards, schema-align projection, LOPO helpers, attribution evaluation. Chapter 4 presents the same-dataset detection results (contribution C2's metric component). Chapter 5 is the cross-testbed and LOPO study (C1 and C2). Chapter 6 is the attribution study (C3). Chapter 7 discusses reviewer-anticipated objections and limitations. Chapter 8 concludes.

## 1.4 Scope and non-goals

- We do not propose a new detector architecture. The argument is empirical: existing detectors evaluated under a protocol that makes their brittleness visible.
- We do not use HAI 22.04 or 23.05 despite their larger attack sets. 21.03 was already available via git-standard clone and has per-process attack tags sufficient for C3. HAI 22.04 is listed as follow-up.
- We do not evaluate against SWaT or WADI. Those are single-testbed benchmarks like the ones we already use; adding them does not change the argument of C1.
- Defensive deployment and real-time performance are out of scope. Every reported number is from offline batched evaluation.
