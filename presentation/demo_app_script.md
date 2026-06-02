# 90-Second Web-App Demo — talking points

Open **http://127.0.0.1:8000/**. Every beat is: pick an **Artifact**, pick a **Dataset**,
click **Score**. Read the **bold**. ~200 words at ~140 wpm.

> Framing rule: never say "my web app." Say "I trained a detector on one testbed and
> scored another — watch what happens." The app is just the screen the finding happens on.

---

## Which sample to use with which artifact (verified numbers)

| Beat | Artifact (dropdown) | Dataset (dropdown) | Result on screen |
|---|---|---|---|
| 1 native | `baseline_hai_isolation_forest/seed42` | `hai_test_sample` | pointwise F1 **0.21** · point-adjust F1 **0.81** |
| 2 collapse | `transfer_morris_to_hai_isolation_forest/seed42` | `hai_test_sample` *(same file as beat 1)* | F1 **0.06**, recall **1.00**, precision **0.032** |
| 3 guard | `baseline_morris_isolation_forest/seed42` | `hai_test_sample` | **HTTP 400** — schema mismatch, lists missing/unexpected columns |

The two demo datasets:
- **`hai_test_sample`** — HAI 21.03 slice, 15 000 rows, 79 sensors, 3.2 % attacks. Use with HAI + transfer artifacts.
- **`morris_gas_test_sample`** — Morris gas pipeline, 15 000 rows, 16 features, 48.6 % attacks. (Not needed for this 90 s cut.)

---

## Shot list + narration

### Beat 1 — A detector on its own testbed (0:00–0:30)
Select `baseline_hai_isolation_forest/seed42` + `hai_test_sample` → **Score**.

**Say:** "This is an anomaly detector trained and tested on the same plant — HAI, a Korean
power-plant testbed. Here's the catch you can already see: under strict point-wise F1 it
scores 0.21, but under the point-adjust metric the field over-uses, the *same* result jumps
to 0.81. Same model, same data — the metric alone quadruples the score. That's our second
finding, on screen."

### Beat 2 — The headline: cross-testbed collapse (0:30–1:05)
Keep `hai_test_sample`. Change **only** the artifact to
`transfer_morris_to_hai_isolation_forest/seed42` → **Score**.

**Say:** "Now I change one thing — the model. Same HAI file, but this detector was trained
on a completely different testbed, a gas pipeline, and ported across. Watch it collapse.
F1 drops to 0.06, recall is 1.0, precision is 0.032 — it just flags *everything*, and that
0.032 is exactly the dataset's attack rate. It isn't detecting; it's guessing. That's our
core finding: detectors that look strong at home don't transfer across plants."

### Beat 3 — It fails safe (1:05–1:25)
Keep `hai_test_sample`. Change the artifact to `baseline_morris_isolation_forest/seed42`
→ **Score**.

**Say:** "And if you point a Morris-trained model at HAI data outright, it doesn't silently
mis-score — it refuses, with a schema mismatch and the exact columns that don't match.
The pipeline fails loud, not wrong."

### Beat 4 — Close (1:25–1:30)
**Say:** "Same code, same input — only the training source changed. That's the whole story."

---

## Setup checklist (off camera)
- Server already running: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
- Browser at http://127.0.0.1:8000/ , zoom ~125 % so metrics are legible at 1080p.
- Pre-open the artifact dropdown once so you know where the three entries are; beats 2–3
  are just changing that one field.
- Record 1080p+. Don't resize/close the terminal running uvicorn mid-take.

## If a take goes wrong
Every beat is idempotent — just re-pick and click Score again. No state to reset.
Numbers are deterministic (seed 42), so they'll be identical every run.
