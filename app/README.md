# External-Validation Web App

Thin FastAPI wrapper over `src.inference` for scoring unseen Morris gas-pipeline ARFF/CSV files against a saved model artifact.

## Run

```bash
# 1. Train at least one artifact so there is something to select.
# Add artifact.save_dir to any YAML in experiments/configs/, then:
python -m experiments.run experiments/configs/<your_config>.yaml

# 2. Launch the app:
uvicorn app.main:app --reload
# Open http://localhost:8000
```

## Endpoints

- `GET /artifacts` — list saved artifacts discovered under `results/checkpoints/`.
- `GET /demo-datasets` — list bundled demo datasets discovered under `data/demo/manifest.json`.
- `POST /score` — multipart form fields:
  - `artifact_id` (required)
  - **exactly one** of `file` (`.arff` or `.csv` upload) or `bundled_dataset` (id from `/demo-datasets`)
  Returns JSON `{artifact, n_input_rows, n_scored, n_flagged, threshold, metrics, preview, download_url}`.
- `GET /downloads/{run_id}/scores.parquet` — download the full per-row scores + flags (+ labels when present).
- `GET /docs` — FastAPI-generated OpenAPI page.

## Bundled demo datasets

The app ships with two test-split slices under `data/demo/` so visitors can score the canonical thesis datasets in one click without acquiring the source data themselves:

- **`hai_test_sample.csv`** — ~15k contiguous rows from the HAI 21.03 test split with both attack and normal rows. Native sensor column names (`P1_*`, `P2_*`, `P3_*`, `P4_*`) preserved. Use with HAI-trained artifacts.
- **`morris_gas_test_sample.csv`** — ~15k contiguous rows from the Morris gas-pipeline test split. `IanArffDataset.arff` schema preserved. Use with Morris-trained or HAI→Morris transfer artifacts.

Both are produced deterministically (seed=42, first contiguous test-window of length ≤ 15000 containing both classes) by `python -m scripts.build_demo_datasets`. Re-running yields byte-identical files. The manifest at `data/demo/manifest.json` is the source of truth for `/demo-datasets`.

## Scope and caveats

- **Localhost only.** No authentication, no TLS. Do not expose this to a network.
- **50 MB upload cap.** Enforced server-side while streaming the upload.
- **Whole file in memory.** Morris gas-pipeline captures are well under this cap; larger datasets would need a streaming rewrite.
- **Threshold is frozen from training.** The saved artifact's threshold is applied as-is. Scoring a cross-testbed file with a source-fitted threshold is a known limitation of the transfer setting — see `thesis/chapters/07_discussion.tex` for the discussion of threshold-transfer semantics.
- **Scaler is not re-validated.** The MinMaxScaler was fit on the training dataset's normal split. For a new file we assume the user is providing evaluation data only; no leak check is re-performed.

## Adapter scope

- **`morris_gas`** — ARFF/CSV with the canonical `IanArffDataset.arff` column conventions (`binary result` label, fixed feature names). Validates exact column match against the artifact's `feature_columns`.
