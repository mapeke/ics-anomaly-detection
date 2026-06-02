# Running the Web Demo App from a Fresh Clone

Step-by-step, starting from nothing but a GitHub checkout. Windows/PowerShell commands
shown; macOS/Linux equivalents in parentheses.

> **Good news:** the repo ships both the demo **datasets** (`data/demo/`) **and** the three
> trained model artifacts the demo uses (`results/checkpoints/`, seed42). So the fast path is
> just clone → install → launch — no data download, no training. Budget ~5 minutes.
>
> Steps 3–4 (raw data + training) are **optional** — only needed if you want to retrain or
> add other models/seeds. Skip straight from Step 2 to Step 5 for the demo.

---

## 0. Prerequisites

- **Python 3.11 or 3.12** (`python --version`)
- **git**
- ~2 GB free disk (raw HAI clone is the bulk)
- No GPU needed — the demo uses Isolation Forest (scikit-learn, CPU).

## 1. Clone the repo

```powershell
git clone https://github.com/mapeke/ics-anomaly-detection.git
cd ics-anomaly-detection
```

## 2. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # (macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run once:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

## 3. (Optional) Get the raw datasets

> Skip this and Step 4 for the demo — the trained artifacts already ship. Do this only to
> retrain or add other models/seeds.

The demo's three models are trained on HAI and Morris. Fetch both into `data/raw/`.

**HAI 21.03** (plain clone):
```powershell
cd data/raw
git clone --depth 1 https://github.com/icsdataset/hai.git hai
cd ../..
```
Expected: `data/raw/hai/hai-21.03/train1.csv.gz`, `test1.csv.gz`, etc.

**Morris gas pipeline** (manual download — no direct git URL):
1. Open https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
2. Download the gas-pipeline ARFF archive.
3. Extract so the file lands at `data/raw/morris/IanArffDataset.arff`.

(Full details and attribution: `data/README.md`.)

## 4. (Optional) Retrain the artifacts

> Skip for the demo — the three artifacts already ship at seed42. Do this only to
> regenerate them or produce other models/seeds. Each command trains 3 seeds and writes the
> artifacts + result rows.

```powershell
python -m experiments.run experiments/configs/baseline_hai_isolation_forest.yaml
python -m experiments.run experiments/configs/baseline_morris_isolation_forest.yaml
python -m experiments.run_transfer experiments/configs/transfer_morris_to_hai_isolation_forest.yaml
```

- Note the **`run_transfer`** driver on the third line — transfer configs use it, not `run`.
- Each takes ~1–2 minutes on a laptop (Isolation Forest is fast).
- Verify they landed:
  ```powershell
  python -c "from pathlib import Path; print(*sorted(p.name for p in Path('results/checkpoints').iterdir()), sep='\n')"
  ```
  You should see `baseline_hai_isolation_forest`, `baseline_morris_isolation_forest`,
  `transfer_morris_to_hai_isolation_forest`.

## 5. Start the server

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Leave this terminal running. Open a browser at:

- **UI:** http://127.0.0.1:8000/
- **API docs (Swagger):** http://127.0.0.1:8000/docs

> Localhost only — the app has no auth. Don't bind to `0.0.0.0` or expose it to a network.

## 6. Run the demo

In the UI, each run is: pick an **Artifact**, pick a **Dataset**, click **Score**.
Verified results with the shipped demo datasets:

| Artifact | Dataset | Expected |
|---|---|---|
| `baseline_hai_isolation_forest/seed42` | `hai_test_sample` | pointwise F1 ≈ 0.21, point-adjust F1 ≈ 0.81 |
| `transfer_morris_to_hai_isolation_forest/seed42` | `hai_test_sample` | F1 ≈ 0.06, recall 1.00, precision 0.032 (collapse) |
| `baseline_morris_isolation_forest/seed42` | `morris_gas_test_sample` | pointwise F1 ≈ 0.08, P 0.91 |
| `baseline_morris_isolation_forest/seed42` | `hai_test_sample` | **HTTP 400** schema mismatch (expected — wrong dataset for the model) |

## 7. Stop the server

`Ctrl+C` in the uvicorn terminal. If you lost the terminal:
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/artifacts` returns an empty list | Step 4 didn't run, or wrong working dir. Re-run the training commands from the repo root. |
| `FileNotFoundError` for `hai-21.03` / `.arff` during training | Raw data missing or misplaced — recheck Step 3 paths against `data/README.md`. |
| `ModuleNotFoundError: fastapi` (or others) | venv not activated, or `pip install -r requirements.txt` skipped. |
| `git clone` of HAI fails with LFS smudge errors | You're pulling a newer release (22.04/23.05) — the project uses **21.03**, which is plain gzip and clones fine. |
| Port 8000 already in use | Stop the old server (Step 7) or start with `--port 8001`. |
| PowerShell won't run `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then retry. |

## What ships vs. what doesn't

- **Ships (run immediately):** the three seed42 demo artifacts under `results/checkpoints/`
  and the demo CSVs under `data/demo/`. Scoring needs only these.
- **Doesn't ship (Steps 3–4):** raw datasets (`data/raw/`, ~625 MB, gitignored) and all other
  models/seeds. Needed only to retrain or extend the benchmark — never for the demo itself.
