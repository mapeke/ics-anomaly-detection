# ICS Anomaly Detection for Cyber-Physical Security

Undergraduate diploma project comparing four anomaly-detection models on two real-world Industrial Control System (ICS) datasets.

## Datasets

| Dataset | Domain | Source |
|---|---|---|
| **HAI 23.05** | Boiler / turbine / water-treatment HIL testbed | [icsdataset/hai](https://github.com/icsdataset/hai) |
| **Morris Gas Pipeline** | Natural gas pipeline Modbus RTU | [tommy-morris-uah](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) |

See [`data/README.md`](data/README.md) for retrieval instructions.

## Models

| Model | Family | File |
|---|---|---|
| Isolation Forest | Tree ensemble, unsupervised | `src/models/isolation_forest.py` |
| One-Class SVM | Kernel, unsupervised | `src/models/ocsvm.py` |
| Dense Autoencoder | Neural, unsupervised | `src/models/autoencoder.py` |
| LSTM Autoencoder | Recurrent neural, unsupervised | `src/models/lstm_autoencoder.py` |

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt

# Fetch datasets (see data/README.md)
# Then run notebooks in order
jupyter lab notebooks/
```

## Notebooks

1. `01_data_exploration.ipynb` — sensor distributions, attack-window visualisation.
2. `02_preprocessing.ipynb` — scaling, windowing, train/val/test splits.
3. `03_classical_models.ipynb` — Isolation Forest, One-Class SVM.
4. `04_deep_models.ipynb` — Dense and LSTM autoencoders.
5. `05_results_comparison.ipynb` — headline metrics, bar charts, ROC/PR curves.

## Results

After running all notebooks:

- Aggregated metrics: `results/metrics/summary.csv`
- Figures for the thesis: `results/figures/`

## Project Plan

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full methodology and verification steps.

## License

Code: MIT. Datasets retain their original licenses.
