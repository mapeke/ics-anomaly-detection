# 8. Conclusion

We benchmarked six ICS anomaly detectors — three classical, three windowed SOTA — on HAI 21.03 and the Morris gas-pipeline dataset under three time-aware metrics, evaluated cross-testbed transfer under two calibration regimes with a within-HAI leave-one-process-out control, and evaluated per-sensor attribution at the process level. Three findings define the thesis:

1. **Source-calibrated cross-testbed transfer is a class-prior artifact.** Under the common practice of fixing the threshold from source-validation data, every detector degenerates to the predict-all trivial baseline of the target's class prior. Target-calibration reveals a meaningful classical-vs-SOTA separation invisible to the source-calibrated protocol.

2. **Feature shift produces targeted, not global, detection blindness.** Within-HAI LOPO shows that dropping the turbine-controller features specifically blinds detectors to turbine-targeted attacks (from F1 0.41 to 0.03); dropping the water-loop features globally degrades detection because they carry the bulk of the "normal" signal. Sensor-level transfer within one testbed tops out at ~0.57 pointwise F1 — an order of magnitude better than cross-testbed, a useful upper bound.

3. **Detection and localization are a genuine tradeoff.** The weakest detector (Dense AE) is the only model whose per-feature attribution beats a random baseline on all three attacked processes. Windowed SOTA models concentrate attribution on the majority-attacked process regardless of actual target, and attention-weighted rollout for TranAD provides no additional localization signal. For an operator triaging a flagged window, Dense AE's output is the more actionable.

The single recommendation this thesis makes to the ICS anomaly detection community is to report the full grid — multiple metrics, multiple calibration regimes, attribution alongside detection — rather than the single headline F1 that currently dominates publications. The protocol and infrastructure to do so are contributed in this thesis's `src/` and `experiments/`, all driven by committed YAML configurations. Every number in the document can be reproduced with `python -m experiments.run[_transfer|_lopo|_attribution] <config.yaml>`.

Future work (§7.5) includes per-sensor attribution on HAI 22.04, multi-seed attribution confirmation, and a learned-projection variant of the schema alignment.
