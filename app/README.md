# Demo Web App — How to Run

Thin FastAPI wrapper over `src.inference`. Loads a trained model artifact, accepts an uploaded ICS sensor file (or a bundled sample), runs the same scoring pipeline used in the experiments, and returns metrics + per-row anomaly scores.

The app is what backs the cross-dataset story in the thesis: same artifact, same input format, swap the artifact's `trained_on` and watch the metrics collapse.

## Prerequisites

1. Python deps installed:
   ```powershell
   pip install -r requirements.txt
   ```
2. At least one model artifact saved under `results/checkpoints/`. Three ship with the repo (seed42) so the app runs straight after a clone — `baseline_hai_isolation_forest`, `baseline_morris_isolation_forest`, and `transfer_morris_to_hai_isolation_forest`. To produce more (other models/seeds), run an experiment yourself:
   ```powershell
   python -m experiments.run experiments/configs/baseline_morris_isolation_forest.yaml
   ```
3. (Optional) bundled demo CSVs at `data/demo/`. Already present; regenerate with `python -m scripts.build_demo_datasets` if needed.

## Run

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then:

- **UI**: <http://127.0.0.1:8000/>
- **Swagger / OpenAPI**: <http://127.0.0.1:8000/docs>

Add `--reload` while developing. Don't bind to `0.0.0.0` — the app has no auth.

## Demo scenarios

Each row is one click in the UI. The expected metrics below are what the committed artifacts produce on the bundled datasets — useful for a defense walkthrough.

| # | Story | Upload | Artifact | Expected |
|---|---|---|---|---|
| 1 | Native model on its own data | `morris_gas_test_sample.csv` | `baseline_morris_isolation_forest/seed42` | P=0.910 R=0.043 F1=0.082 |
| 2 | Native model on its own data | `hai_test_sample.csv` | `baseline_hai_isolation_forest/seed42`    | P=0.455 R=0.135 F1=0.209 |
| 3 | **Cross-dataset transfer collapses** | `hai_test_sample.csv` | `transfer_morris_to_hai_isolation_forest/seed42` | P=0.032 R=1.0 F1=0.062 — model flags every row, precision = the dataset's base attack rate |
| 4 | Schema mismatch (input validation) | `hai_test_sample.csv` | `baseline_morris_isolation_forest/seed42` | 400 with `missing` / `unexpected` column lists |

Scenarios 1+3 share an input file. Only the artifact's `trained_on` differs — that's the whole point.

Files for the upload path are at `presentation/hai_test_sample.csv` and `presentation/morris_gas_test_sample.csv` (copies of the bundled datasets, in a folder that's easy to browse to from the file picker).

## How input is routed

The artifact tells the app what to expect. There is no "source dataset" form field — it's all inferred from the saved manifest:

1. **Adapter dispatch** — `expected_input_kind(artifact)` reads `manifest.extra.target_dataset` (falling back to parsing `trained_on`). Returns `"hai"` or `"morris"`. The matching adapter (`src/inference/adapters/{hai,morris_gas}.py`) reads the file and returns clean features.
2. **Optional projection** — if the artifact was trained on canonical type-vectors (any `*__to__*` transfer artifact), `schema_align.project_dataframe` collapses the raw input to the 6 canonical types using `data/feature_types.yaml` before scoring.
3. **Score** — `src.inference.score_dataframe` applies the artifact's scaler, windows if required, calls `model.score`, then evaluates pointwise / point-adjust / eTaPR if labels are present.

So a `transfer_morris_to_hai_*` artifact paired with raw HAI data: adapter loads HAI columns, projection collapses 79 sensors → 6 types, model scores. A `baseline_morris_*` artifact with the same raw HAI data: the dispatcher sees `target_dataset=morris`, the Morris adapter rejects it with a schema-mismatch error.

## Endpoints

- `GET /artifacts` — list saved artifacts under `results/checkpoints/`. Each entry has the model name, `trained_on`, feature count, threshold, config hash, seed, and git SHA from training.
- `GET /demo-datasets` — list bundled samples under `data/demo/`.
- `POST /score` (multipart) — fields:
  - `artifact_id` (required)
  - **exactly one of**: `file` (.arff or .csv upload, ≤ 50 MB) or `bundled_dataset` (id from `/demo-datasets`)
- `GET /downloads/{run_id}/scores.parquet` — full per-row score + flag (+ label, if the input had one).
- `GET /docs` — FastAPI's auto-generated Swagger UI.

## Score sign convention

`score(x) = -decision_function(x)` for IF / OCSVM, reconstruction error for AE-family models. Either way: **higher = more anomalous**. The preview table shows raw scores, so negative values for IF/OCSVM mean "the model thinks this row is normal." `flag = score >= threshold` is the binary decision.

## Adapters

- **`hai`** — CSVs with HAI 21.03 sensor columns. Accepts both the raw release files (per-process `attack*` flag columns) and the cleaned demo slice (single `label` column). Schema normalisation goes through `src.data_loader.prepare_hai_frame` so it's bit-identical to training.
- **`morris_gas`** — ARFF or CSV with `IanArffDataset.arff` conventions (`binary result` label or `label`, the standard 16 feature columns). Goes through `src.data_loader.prepare_morris_frame`.

Both adapters raise `SchemaMismatchError` (mapped to HTTP 400 with explicit `missing` / `unexpected` arrays) when the upload's columns don't match what the artifact expects.

## Scope and caveats

- **Localhost only.** No auth, no TLS. Don't expose to a network.
- **50 MB upload cap**, enforced while streaming.
- **Whole file in memory.** Fine for HAI 21.03 / Morris; larger datasets would need a streaming rewrite.
- **Threshold is frozen from training.** Cross-testbed scoring uses the source-fitted threshold; the discussion in `thesis/chapters/07_discussion.tex` explains why this is a deliberate methodological choice.
- **Scaler not re-validated** at inference. We assume the user provides evaluation data; no leak check is re-performed.

## Stop the server

`Ctrl+C` in the terminal running uvicorn. If you started it as a background task and lost the handle:

```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```
