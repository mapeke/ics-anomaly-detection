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
- `GET /variants` — list pre-baked dataset variants under `data/feature_types_variants/` (Phase B).
- `POST /score` — multipart form fields:
  - `artifact_id` (required)
  - `file` (required) — `.arff` or `.csv`
  - `adapter` — `morris_gas` (default) or `generic_arff` (Phase B)
  - `variant` — required when `adapter=generic_arff`; the YAML stem from `/variants` (e.g. `morris_gas_final`)
  - `recalibrate` — `target_val_percentile` to recompute the threshold from the uploaded data's normal-only rows; default keeps the artifact's source threshold
  - `percentile` — percentile for `recalibrate` (default 99)
  Returns JSON `{artifact, n_input_rows, n_scored, n_flagged, threshold, source_threshold, recalibrate_mode, recalibrate_percentile, metrics, preview, download_url}`.
- `GET /downloads/{run_id}/scores.parquet` — download the full per-row scores + flags (+ labels when present).
- `GET /docs` — FastAPI-generated OpenAPI page.

## Scope and caveats

- **Localhost only.** No authentication, no TLS. Do not expose this to a network.
- **50 MB upload cap.** Enforced server-side while streaming the upload.
- **Whole file in memory.** Morris gas-pipeline captures are well under this cap; larger datasets would need a streaming rewrite.
- **Threshold is frozen from training.** The saved artifact's threshold is applied as-is. Scoring a cross-testbed file with a source-fitted threshold is a known limitation of the transfer setting — see `thesis/chapters/07_discussion.tex` for the discussion of threshold-transfer semantics.
- **Scaler is not re-validated.** The MinMaxScaler was fit on the training dataset's normal split. For a new file we assume the user is providing evaluation data only; no leak check is re-performed.

## Adapter scope

- **`morris_gas`** (Phase A) — ARFF/CSV with the canonical `IanArffDataset.arff` column conventions (`binary result` label, fixed feature names). Validates exact column match against the artifact's `feature_columns`.
- **`generic_arff`** (Phase B) — accepts any tabular ARFF/CSV plus a variant YAML (under `data/feature_types_variants/`) that maps each source column to a canonical type. Projects via `src/transfer/schema_align.py:project_dataframe` into the artifact's canonical-type feature space. Designed for transfer-trained artifacts (artifacts whose `feature_columns` are canonical types like `pressure`, `pump_state`, etc.). Currently shipped variant: `morris_gas_final` (multiclass-labelled `gas_final.arff` capture).

Power-system adapter and a user-uploaded ad-hoc variant YAML are Phase C; see `C:/Users/user/.claude/plans/can-we-make-an-jaunty-hopper.md`.

## Adding a new variant

1. Create `data/feature_types_variants/<my_variant>.yaml` modelled after `morris_gas_final.yaml`. List `label_column`, `label_semantics`, `drop_columns`, per-column `feature_types`, and per-type `aggregations`.
2. Restart the app — `/variants` re-discovers YAML files at request time.
