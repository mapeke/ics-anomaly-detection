# Project Plan — ICS Anomaly Detection for Cyber-Physical Security

## Context

This is an undergraduate diploma project titled **"Industrial Control System Anomaly Detection for Cyber-Physical Security."** The goal is to evaluate multiple machine learning models on two real-world ICS datasets and present the findings in a reproducible Jupyter notebook workflow suitable for a thesis defense.

## Datasets (2)

1. **HAI 21.03** — HIL-based Augmented ICS Security Dataset (ETRI).
   - Source: https://github.com/icsdataset/hai
   - Characteristics: multivariate time series from boiler, turbine, water-treatment, and HIL simulator processes; labeled attack windows across three sub-processes (`attack_P1/P2/P3`).
   - License: CC BY 4.0 (public).
   - Note: defaulting to 21.03 because 22.04/23.05 require Git LFS and the upstream LFS budget is exhausted. 21.03 is the version most widely cited in academic literature.
2. **Morris Gas Pipeline Dataset** — Mississippi State University.
   - Source: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
   - Characteristics: Modbus RTU telemetry from a gas pipeline testbed; labeled command-injection, response-injection, and reconnaissance attacks.
   - License: public research use.

## Models (4)

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | Isolation Forest | Classical unsupervised | Tree-based, fast, strong tabular baseline |
| 2 | One-Class SVM | Classical unsupervised | RBF kernel, boundary around normal data |
| 3 | Autoencoder (Dense) | Deep unsupervised | Reconstruction error as anomaly score |
| 4 | LSTM Autoencoder | Deep unsupervised, temporal | Sequence reconstruction — suited to ICS time series |

## Repo Structure

```
ics-anomaly-detection/
├── CLAUDE.md                     # assistant memory / project context
├── PROJECT_PLAN.md               # this file
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                      # original downloads (gitignored when large)
│   ├── processed/                # scaled, windowed feather/parquet
│   └── README.md                 # how to obtain each dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # load_hai(), load_morris()
│   ├── preprocessing.py          # scale, sliding windows, train/test split
│   ├── evaluation.py             # precision, recall, F1, ROC, confusion matrix, plots
│   ├── utils.py                  # seeding, paths
│   └── models/
│       ├── __init__.py
│       ├── isolation_forest.py
│       ├── ocsvm.py
│       ├── autoencoder.py
│       └── lstm_autoencoder.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_deep_models.ipynb
│   └── 05_results_comparison.ipynb
└── results/
    ├── figures/                  # .png plots
    └── metrics/                  # .csv metric tables
```

## Methodology

1. **Exploration** — distributions, correlations, sensor-wise plots, attack-window visualization.
2. **Preprocessing** — min-max scaling fitted on normal data only; sliding windows for LSTM; 70/15/15 split on normal data; all attack rows go to the test set.
3. **Training** — unsupervised on normal data; thresholds tuned on validation set via best-F1 sweep.
4. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix per model × dataset.
5. **Comparison** — side-by-side bar charts and LaTeX-ready metric tables.

## Deliverables

- Reproducible Jupyter notebooks with narrative, code, plots, metrics.
- `results/figures/` and `results/metrics/` committed for thesis inclusion.
- Public GitHub repo: `mapeke/ics-anomaly-detection`.

## Verification

- `python -c "import src.data_loader as d; d.load_hai(); d.load_morris()"` loads both datasets.
- Each notebook runs top-to-bottom with no errors (`jupyter nbconvert --execute`).
- `results/metrics/summary.csv` contains one row per (model, dataset) pair with all metrics populated.

## Critical Files

- `src/data_loader.py` — dataset loaders, must handle HAI's multi-CSV layout and Morris's ARFF format.
- `src/preprocessing.py` — sliding-window utility shared by LSTM and classical pipelines.
- `src/models/lstm_autoencoder.py` — Keras sequence AE; GPU optional.
- `notebooks/05_results_comparison.ipynb` — the headline artifact for the thesis defense.
