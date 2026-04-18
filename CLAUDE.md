# CLAUDE.md

Context for Claude Code when working on this repository.

## Project

Undergraduate diploma project — **"Industrial Control System Anomaly Detection for Cyber-Physical Security."** See `PROJECT_PLAN.md` for the detailed plan.

- Student: Daniyal
- Language: Python 3.12
- Platform: Windows 10 (bash shell available via Git Bash)
- GitHub: `mapeke/ics-anomaly-detection` (public)

## Datasets

1. **HAI 23.05** — `data/raw/hai/` — ETRI HIL-based ICS dataset. Multivariate time series, labeled attacks. Cloned from https://github.com/icsdataset/hai (CC BY 4.0).
2. **Morris Gas Pipeline** — `data/raw/morris/` — Mississippi State gas pipeline dataset. ARFF format, labeled command/response injection attacks. From https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets.

Raw data is large; keep it gitignored and document retrieval in `data/README.md`.

## Models

Four models compared across both datasets:
- Isolation Forest (`src/models/isolation_forest.py`)
- One-Class SVM (`src/models/ocsvm.py`)
- Dense Autoencoder (`src/models/autoencoder.py`)
- LSTM Autoencoder (`src/models/lstm_autoencoder.py`)

All trained unsupervised on normal data only. Anomaly thresholds tuned on a validation split.

## Conventions

- Use `src.utils.set_seed(42)` at the top of every notebook for reproducibility.
- Figures saved to `results/figures/<notebook>_<name>.png` at 150 dpi.
- Metric tables written to `results/metrics/<model>_<dataset>.csv` and aggregated in `summary.csv`.
- Never commit `data/raw/` (gitignored). `data/processed/` is small enough to commit as Parquet.

## How to run

```bash
pip install -r requirements.txt
jupyter lab
# then execute notebooks/01 → 05 in order
```

## Commands

- Lint: none configured; rely on clean code.
- Tests: none — notebooks serve as the functional test surface.
- Execute all: `jupyter nbconvert --to notebook --execute notebooks/*.ipynb`

## Workflow

- Commit after each logical step (scaffolding, each module, each notebook).
- Commit message style: imperative, short subject line, optional body.
- Push after every commit so the thesis advisor can track progress on GitHub.
