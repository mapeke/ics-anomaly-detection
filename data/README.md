# Datasets

Both datasets are **not** checked into git (see repo `.gitignore`). Retrieve them locally before running any notebook.

## 1. HAI 23.05

Official repo: https://github.com/icsdataset/hai (CC BY 4.0)

```bash
cd data/raw
git clone --depth 1 https://github.com/icsdataset/hai.git hai
```

Expected layout after cloning:

```
data/raw/hai/
├── hai-23.05/
│   ├── train1.csv
│   ├── train2.csv
│   ├── ...
│   └── test*.csv
├── hai-22.04/
└── README.md
```

Our loader (`src/data_loader.py::load_hai`) uses the `hai-23.05` release by default.

## 2. Morris Gas Pipeline

Landing page: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets

The canonical file is `IanArffDataset.arff` (combined gas-pipeline + water-storage). For this project we want **gas pipeline only** — use `Gas_Final.arff` if available, otherwise filter `IanArffDataset.arff` by the `system` column.

Manual download steps:
1. Visit the landing page above.
2. Download the gas-pipeline ARFF archive.
3. Extract to `data/raw/morris/`.

Expected layout:

```
data/raw/morris/
└── IanArffDataset.arff   (or Gas_Final.arff)
```

Our loader (`src/data_loader.py::load_morris`) parses ARFF via `liac-arff`.

## Attribution

- HAI: Shin, H.-K. et al., *HAI 1.0: HIL-based Augmented ICS Security Dataset*, USENIX CSET 2020.
- Morris: Morris, T. et al., *Industrial Control System Traffic Data Sets for Intrusion Detection Research*, 2014.
