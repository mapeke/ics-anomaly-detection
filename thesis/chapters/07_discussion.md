# 7. Discussion

## 7.1 What the three findings add up to

Taken together:
- **Detection (Ch 4):** SOTA windowed models beat classical ones on same-dataset eTaPR; rankings change with metric choice; PA-F1 inflation is real.
- **Transfer (Ch 5):** source-calibrated thresholds don't transfer; target-calibrated thresholds reveal a genuine classical/SOTA split; within-HAI LOPO shows feature-specific blindness.
- **Attribution (Ch 6):** Dense AE beats SOTA on localization; attention rollout adds nothing.

The single thesis-level story: **ICS anomaly detection is not a one-number benchmark.** Different metrics, calibration regimes, and attribution tasks all reshuffle the ranking. Any paper that picks one (model, metric, threshold) triple to report has under-constrained its own claim. Reviewers should ask for the full grid; our contribution is a protocol for producing it.

## 7.2 Likely reviewer objections

**"Your transfer projection is lossy."**
True, by design (Ch 5.1). We accepted option 1 (per-variable type tagging) as the simplest defensible scheme. Option 2 (learned projections) and option 3 (within-testbed-only) remained unused, and we report the ceiling-check against within-HAI LOPO: even in the most favourable within-testbed condition, the ceiling is 0.57 pointwise F1 — an order of magnitude better than cross-testbed but still far from same-dataset. The cross-testbed numbers are not artifacts of a bad projection; they are upper-bounded by the problem itself.

**"TranAD reproducibility gap."**
PA-F1 0.46 vs. Tuli 2022's 0.97. Sources enumerated in §4.6: dataset version, subsample, architecture capacity, threshold protocol. Our TranAD still ranks #1 on HAI PA-F1 in our setup — the qualitative claim survives. The thesis does not depend on matching published absolutes; it depends on comparing models under a consistent protocol.

**"USAD collapsed on Morris — selection bias?"**
USAD eTaPR 0.009 on Morris (Ch 4.4) is not cherry-picked; it is what the model's scores produce. The collapse is reported as a finding about USAD's assumptions on discrete data, not hidden.

**"Only one seed for Phase 4."**
Acknowledged; listed as a limitation. Phase 1–2 multi-seed numbers showed eTaPR variance ≤0.02 across seeds for the classical models; attribution seed-variance is likely similar. The P1/P2/P3 ordering is robust to that magnitude of noise.

**"Process-level attribution is weak."**
Yes; HAI 21.03 does not publish per-sensor attack targets in machine-readable form. Ch 6.6 lists this as the primary weakness and points at HAI 22.04 as a follow-up.

## 7.3 Practical implications for deployment

- **Calibrate on target-normal.** Any deployment should collect a clean-operation period from the target plant before activating the detector and calibrate the threshold there. Source-trained percentile thresholds produce trivial classifiers.
- **Report three metrics.** Readers should not trust single-metric rankings. Our recommendation: pointwise F1 for per-timestep claims, PA-F1 for "at least one timestep of the event was flagged" (and flag the inflation), eTaPR for event-level detection with coverage weighting.
- **Consider Dense AE for localization even if a SOTA model is the detector.** A hybrid pipeline — SOTA for flagging, Dense AE for attribution — would outperform either alone on operator-facing triage.

## 7.4 What we deliberately did not test

- New detector architectures. The thesis is empirical, not architectural.
- SWaT / WADI. Adding more single-testbed datasets does not change C1.
- Real-time streaming performance. Every number is offline batched.
- Federated or differentially-private training. Out of scope.

## 7.5 Future work

1. **HAI 22.04 per-sensor attribution.** 22.04's README contains per-attack target-sensor descriptions that can be parsed into a ground-truth mapping. Running Ch 6's eval against that would upgrade the localization claim from process-level to sensor-level.
2. **Multi-seed attribution** across all six models to confirm or refute the LSTM-AE ≡ USAD coincidence (Finding 4.3).
3. **Learned projection for schema alignment** (CLAUDE.md §5.2 option 2) to test whether the cross-testbed ceiling moves under a richer alignment. We expect modest improvement given the LOPO ceiling argument in §7.2, but the experiment is cheap to run.
4. **PA-F1 replacement in the ICS AD literature.** Our data reproduces the Kim 2022 critique on ICS benchmarks; an editorial push to eTaPR as the default would be a small but useful community-level contribution.
