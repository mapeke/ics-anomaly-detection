# 90-Second Terminal Demo — "The realisation"

On-narrative with defense slides 8 (architecture), 10 (software & reproducibility),
11 (implementation key points). Shows the reproducible benchmarking apparatus the
thesis actually describes — **not** the web app (which is not in the thesis or talk).

Total target: **90 seconds.** Narration ~200 words at ~140 wpm. Read the **bold**.

---

## Before you hit record (setup, off camera)

1. Open a clean PowerShell in the project root: `C:\Users\user\Desktop\DiplomaDaniyal`
2. Have these two files open in your editor on a second tab, scrolled to the right spot:
   - `src/models/base.py`  (the `AnomalyDetector` class, lines 15–35)
   - `src/preprocessing.py` (the `scale_bundle` function — the `assert_no_attack_in_train_val()` line)
3. Bump terminal + editor font size so it's legible at 1080p (Ctrl+= a few times).
4. Optional dry run once so the data is warm in OS cache, then clear the screen (`cls`).

## Files involved

| What | File | Why it's on camera |
|---|---|---|
| The experiment config | `experiments/configs/baseline_morris_isolation_forest.yaml` | "every run is a config, no hidden settings" (slide 8) |
| The runner | `experiments/run.py` (invoked, not shown) | one command -> rows on disk (slide 10) |
| The results table | `results/metrics/summary.parquet` | the reproducibility contract |
| Read-back helper | `scripts/show_results.py` | clean grid for the camera |
| The detector interface | `src/models/base.py` | slide 11, left code block |
| The scaler-leak guard | `src/preprocessing.py` | slide 11, right code block |

---

## Shot list + narration

### Beat 1 — The config (0:00–0:15)
**Command:**
```powershell
Get-Content experiments/configs/baseline_morris_isolation_forest.yaml
```
**Say:** "Every experiment in this project is a config file — model, dataset, seeds,
metrics. No hidden settings. This one runs Isolation Forest on the Morris gas-pipeline
testbed, three seeds, three metrics."

### Beat 2 — Run it (0:15–0:50)
**Command:**
```powershell
python -m experiments.run experiments/configs/baseline_morris_isolation_forest.yaml
```
**Say (as it prints):** "One command. It stamps the run with a config hash, fits on
normal data only, and scores under all three time-aware metrics for each seed."

> **EDITING NOTE:** the run takes ~75 s. Do **not** leave dead air — jump-cut or
> 2–4× speed-ramp the wait, with a small caption "training · 3 seeds · ~75 s".
> Land the cut on the final line: **`wrote 9 rows -> summary.parquet`**.

**Say (on that last line):** "Nine rows — three seeds times three metrics — written
straight to the results table."

### Beat 3 — Read it back (0:50–1:05)
**Command:**
```powershell
python scripts/show_results.py baseline_morris_isolation_forest
```
**Say:** "Read back out of the table: the same model scores point-one-two on strict
point-wise F1, but point-three-eight under the over-forgiving point-adjust metric.
Same model, same data — the metric alone almost triples the score. That gap is one of
our findings, and it's reproducible from this one config."

### Beat 4 — The code that makes it trustworthy (1:05–1:30)
Switch to the editor. Show `src/models/base.py` first, then `src/preprocessing.py`.

**Say (on base.py):** "Two pieces of code carry the engineering. This is the interface
every one of the six models obeys — fit, score, attribute. Nothing downstream needs to
know which model it's holding."

**Say (on preprocessing.py, point at the assert line):** "And this assertion is the
guard against the single most common way these benchmarks get inflated — a data leak in
the scaler. If an attack row ever reaches it, the program crashes instead of lying.
Every number you just saw is produced under that guarantee."

*(End on the assert line. Cut.)*

---

## One-line summary if you need a title card
> "One config → one command → a stamped row in the results table — under a leak guard
> that fails loud. That's the realisation."

## Backup if a live run feels risky on camera
Pre-run Beat 2 once before recording. Then in the take, run the command, immediately
hard-cut to the finished output (you already have it), and keep Beats 3–4 live. Honest
— the numbers are identical — and removes the only timing risk.
