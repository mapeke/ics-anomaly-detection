"""Generate the thin Phase-1 notebook set.

Phase-1 notebooks are *consumers*, not *producers* — they read parquet
summaries and DatasetBundle loaders from src/ and render plots. Any
new plot, metric, or model lives in src/, not here.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def write(path: Path, notebook: dict) -> None:
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 01 — Data exploration (thin)
# ---------------------------------------------------------------------------

nb01 = nb([
    md("# 01 — Data exploration\n\n"
       "Characterise HAI and Morris as DatasetBundles. All logic lives in "
       "`src/data_loader.py`; this notebook only describes and plots.\n"),
    code(
        "import sys; from pathlib import Path\n"
        "sys.path.insert(0, str(Path().resolve().parent))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from src.data_loader import load_hai, load_morris\n"
        "from src.utils import save_figure, set_seed\n"
        "\n"
        "set_seed(42)\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
    ),
    md("## Bundles"),
    code(
        "hai = load_hai()\n"
        "morris = load_morris()\n"
        "\n"
        "summary = pd.DataFrame([{\n"
        "    'dataset': 'HAI 21.03',\n"
        "    'rows': len(hai.labels),\n"
        "    'features': hai.features.shape[1],\n"
        "    'train': int((hai.split == 'train').sum()),\n"
        "    'val':   int((hai.split == 'val').sum()),\n"
        "    'test':  int((hai.split == 'test').sum()),\n"
        "    'attack_rate_test': float(hai.y('test').mean()),\n"
        "}, {\n"
        "    'dataset': 'Morris gas',\n"
        "    'rows': len(morris.labels),\n"
        "    'features': morris.features.shape[1],\n"
        "    'train': int((morris.split == 'train').sum()),\n"
        "    'val':   int((morris.split == 'val').sum()),\n"
        "    'test':  int((morris.split == 'test').sum()),\n"
        "    'attack_rate_test': float(morris.y('test').mean()),\n"
        "}])\n"
        "summary\n"
    ),
    md("## HAI attack-type breakdown"),
    code(
        "mask = hai.mask('test') & (hai.labels > 0)\n"
        "ac = pd.Series(hai.attack_ids[mask]).value_counts()\n"
        "fig, ax = plt.subplots(figsize=(6, 3))\n"
        "ac.plot(kind='bar', ax=ax, color='#e34a33')\n"
        "ax.set_title('HAI test split — attack counts by per-process flag')\n"
        "ax.set_ylabel('rows')\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'hai_attack_breakdown', subdir='01_exploration')\n"
        "plt.show()\n"
    ),
    md("## Sensor trace with attack windows"),
    code(
        "col = hai.features.columns[0]\n"
        "series = hai.features[col].to_numpy()\n"
        "is_test = hai.mask('test')\n"
        "series = series[is_test][:60000]\n"
        "labels = hai.labels[is_test][:60000]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(12, 3))\n"
        "ax.plot(series, color='#2b8cbe', linewidth=0.5, label=col)\n"
        "ax.fill_between(\n"
        "    np.arange(len(series)), series.min(), series.max(),\n"
        "    where=labels.astype(bool), color='#de2d26', alpha=0.25, label='attack',\n"
        ")\n"
        "ax.set_title(f'HAI test — {col} with attack windows')\n"
        "ax.legend(loc='upper right')\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'hai_attack_windows', subdir='01_exploration')\n"
        "plt.show()\n"
    ),
    md("## Morris feature correlation"),
    code(
        "corr = morris.features.corr().fillna(0)\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "sns.heatmap(corr, cmap='RdBu_r', center=0, square=True, ax=ax,\n"
        "            xticklabels=True, yticklabels=True, cbar_kws={'shrink': 0.6})\n"
        "ax.set_title('Morris feature correlation')\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'morris_correlation', subdir='01_exploration')\n"
        "plt.show()\n"
    ),
])


# ---------------------------------------------------------------------------
# 02 — Preprocessing sanity
# ---------------------------------------------------------------------------

nb02 = nb([
    md("# 02 — Preprocessing sanity\n\n"
       "Verify the scaler fits on train-only and that windowing produces expected\n"
       "shapes. Real preprocessing happens inside `experiments/run.py`.\n"),
    code(
        "import sys; from pathlib import Path\n"
        "sys.path.insert(0, str(Path().resolve().parent))\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from src.data_loader import load_hai, load_morris\n"
        "from src.preprocessing import scale_bundle, make_windows\n"
        "from src.utils import save_figure, set_seed\n"
        "\n"
        "set_seed(42)\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
    ),
    code(
        "hai = load_hai()\n"
        "hai_scaled = scale_bundle(hai)\n"
        "print('HAI splits:', hai_scaled.X_train.shape, hai_scaled.X_val.shape, hai_scaled.X_test.shape)\n"
        "\n"
        "morris = load_morris()\n"
        "morris_scaled = scale_bundle(morris)\n"
        "print('Morris splits:', morris_scaled.X_train.shape, morris_scaled.X_val.shape, morris_scaled.X_test.shape)\n"
    ),
    code(
        "# Scaling effect on a single HAI sensor.\n"
        "col_idx = 0\n"
        "raw = hai.features.iloc[:, col_idx].to_numpy()[hai.mask('train')][:5000]\n"
        "scaled = hai_scaled.X_train[:5000, col_idx]\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
        "axes[0].plot(raw, color='#2b8cbe', linewidth=0.6)\n"
        "axes[0].set_title(f'Raw {hai_scaled.feature_names[col_idx]}')\n"
        "axes[1].plot(scaled, color='#31a354', linewidth=0.6)\n"
        "axes[1].set_title(f'MinMax-scaled {hai_scaled.feature_names[col_idx]}')\n"
        "axes[1].set_ylim(-0.05, 1.05)\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'hai_scaling_effect', subdir='02_preprocessing')\n"
        "plt.show()\n"
    ),
    code(
        "windows = make_windows(hai_scaled.X_test, window=60, stride=10)\n"
        "print('HAI test windows:', windows.shape)\n"
        "print('Min-max of scaled training data:', hai_scaled.X_train.min(), hai_scaled.X_train.max())\n"
    ),
])


# ---------------------------------------------------------------------------
# 03 — Baseline results (reads summary.parquet)
# ---------------------------------------------------------------------------

nb03 = nb([
    md("# 03 — Baseline results\n\n"
       "Reads `results/metrics/summary.parquet` produced by `experiments/run.py`\n"
       "and renders comparison plots. To regenerate numbers:\n\n"
       "```bash\n"
       "for cfg in experiments/configs/baseline_*.yaml; do\n"
       "  python -m experiments.run $cfg\n"
       "done\n"
       "```\n"),
    code(
        "import sys; from pathlib import Path\n"
        "sys.path.insert(0, str(Path().resolve().parent))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from src.utils import METRICS_DIR, save_figure\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
    ),
    code(
        "df = pd.read_parquet(METRICS_DIR / 'summary.parquet')\n"
        "print('rows:', len(df), 'configs:', df['run_name'].nunique())\n"
        "df.head()\n"
    ),
    code(
        "# F1 per model × dataset × metric, averaged over seeds.\n"
        "agg = (df.groupby(['dataset', 'model', 'metric'])\n"
        "         .agg(f1_mean=('f1', 'mean'), f1_std=('f1', 'std'), etapr_mean=('etapr_f1', 'mean'))\n"
        "         .reset_index())\n"
        "# For eTaPR rows, pick the etapr_f1 column into the f1 field.\n"
        "agg.loc[agg['metric'] == 'etapr', 'f1_mean'] = agg.loc[agg['metric'] == 'etapr', 'etapr_mean']\n"
        "agg[['dataset', 'model', 'metric', 'f1_mean', 'f1_std']]\n"
    ),
    code(
        "# Headline chart: F1 by metric, grouped by (dataset, model).\n"
        "plot_df = agg.pivot_table(index=['dataset', 'model'], columns='metric', values='f1_mean')\n"
        "plot_df = plot_df.reindex(columns=['pointwise', 'point_adjust', 'etapr'])\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "plot_df.plot(kind='bar', ax=ax, colormap='viridis')\n"
        "ax.set_ylabel('F1')\n"
        "ax.set_ylim(0, 1)\n"
        "ax.set_title('Baseline F1 by evaluation metric')\n"
        "ax.legend(title='metric')\n"
        "plt.xticks(rotation=30, ha='right')\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'baseline_f1_by_metric', subdir='03_baseline')\n"
        "plt.show()\n"
    ),
    md("## Markdown-ready table"),
    code(
        "print(plot_df.round(3).to_markdown())\n"
    ),
])


# ---------------------------------------------------------------------------
# 06 — Metric sensitivity (Kim 2022 motivation)
# ---------------------------------------------------------------------------

nb06 = nb([
    md("# 06 — Metric sensitivity\n\n"
       "The same model can look strong or weak depending on the evaluation metric.\n"
       "This notebook pulls paired (model, dataset) rows from the parquet summary\n"
       "and shows how point-adjust F1 inflates scores vs. point-wise F1 and eTaPR.\n"),
    code(
        "import sys; from pathlib import Path\n"
        "sys.path.insert(0, str(Path().resolve().parent))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "from src.utils import METRICS_DIR, save_figure\n"
        "sns.set_theme(style='whitegrid', context='notebook')\n"
    ),
    code(
        "df = pd.read_parquet(METRICS_DIR / 'summary.parquet')\n"
        "# Unify F1 column across metrics.\n"
        "df['F1'] = df['f1'].where(df['metric'] != 'etapr', df['etapr_f1'])\n"
        "df.groupby(['dataset', 'model', 'metric'])['F1'].mean().round(3).unstack('metric')\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)\n"
        "for ax, ds in zip(axes, df['dataset'].unique()):\n"
        "    sub = df[df['dataset'] == ds].groupby(['model', 'metric'])['F1'].mean().unstack('metric')\n"
        "    sub = sub.reindex(columns=['pointwise', 'point_adjust', 'etapr'])\n"
        "    sub.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)\n"
        "    ax.set_title(f'{ds} — F1 by metric')\n"
        "    ax.set_ylim(0, 1)\n"
        "plt.tight_layout()\n"
        "save_figure(fig, 'metric_sensitivity', subdir='06_metric_sensitivity')\n"
        "plt.show()\n"
    ),
    md(
        "### Interpretation\n\n"
        "- **Point-adjust F1** systematically inflates scores — any single\n"
        "  detection inside an attack window flips the whole window positive\n"
        "  (Xu et al. 2018). Kim et al. AAAI 2022 demonstrate this gives random\n"
        "  scores ~0.98 PA-F1 on common benchmarks.\n"
        "- **eTaPR** is stricter: it rewards partial overlap proportionally\n"
        "  rather than awarding full credit for one hit.\n"
        "- **Point-wise F1** is the most conservative; deployable systems must\n"
        "  clear this bar rather than the PA bar.\n"
    ),
])


# ---------------------------------------------------------------------------
# 04, 05, 07 — placeholders for later phases
# ---------------------------------------------------------------------------

def placeholder(title: str, phase: str) -> dict:
    return nb([
        md(f"# {title}\n\n**Phase {phase} placeholder.** Filled in when the\n"
           f"upstream src/ code lands (see CLAUDE.md §6 phase plan).\n"),
    ])


write(NOTEBOOKS / "01_data_exploration.ipynb", nb01)
write(NOTEBOOKS / "02_preprocessing_sanity.ipynb", nb02)
write(NOTEBOOKS / "03_baseline_results.ipynb", nb03)
write(NOTEBOOKS / "04_sota_results.ipynb", placeholder("04 — SOTA results", "2"))
write(NOTEBOOKS / "05_cross_dataset.ipynb", placeholder("05 — Cross-dataset study", "3"))
write(NOTEBOOKS / "06_metric_sensitivity.ipynb", nb06)
write(NOTEBOOKS / "07_attribution.ipynb", placeholder("07 — Attribution", "4"))
