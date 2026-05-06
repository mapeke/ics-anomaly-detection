"""Build the bundled demo datasets shipped with the FastAPI app.

Produces two test-split slices that the demo's "Use bundled sample" radio
exposes via ``GET /demo-datasets`` and accepts in ``POST /score``:

- ``data/demo/hai_test_sample.csv``        — raw HAI 21.03 sensor columns
- ``data/demo/morris_gas_test_sample.csv`` — Morris gas-pipeline schema

Plus a manifest at ``data/demo/manifest.json`` enumerating both with
descriptive metadata.

Determinism: same seed (42) as the experiment configs, deterministic
slice rule (first contiguous window of length <= 15000 containing both
attack and normal rows). Re-running this script must produce
byte-identical CSVs and manifest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_hai, load_morris  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"
MAX_ROWS = 15_000
EXPAND_ROWS = 30_000


def first_mixed_window(labels: np.ndarray, max_rows: int = MAX_ROWS) -> tuple[int, int]:
    """Return (start, end) of the first contiguous slice with both classes.

    Walks the test split in non-overlapping ``max_rows``-sized chunks and
    returns the first chunk that contains both an attack row and a normal
    row. If none does, expands the chunk size to ``EXPAND_ROWS`` and tries
    again, then truncates the resulting chunk back to ``max_rows``. Final
    fallback is the first ``max_rows`` rows as-is (with a logged warning).
    """
    n = len(labels)
    for start in range(0, n, max_rows):
        end = min(start + max_rows, n)
        chunk = labels[start:end]
        if 0 in chunk and 1 in chunk:
            return start, end
    for start in range(0, n, EXPAND_ROWS):
        end = min(start + EXPAND_ROWS, n)
        chunk = labels[start:end]
        if 0 in chunk and 1 in chunk:
            return start, min(start + max_rows, end)
    print(
        f"   WARNING: no contiguous window of length <= {EXPAND_ROWS} "
        f"contains both classes; falling back to first {max_rows} rows. "
        f"attack_rate may be skewed."
    )
    return 0, min(max_rows, n)


def slice_dataset(name: str, bundle, label_col: str = "label") -> dict:
    """Slice the bundle's test split, write a CSV, return manifest entry."""
    split_arr = np.asarray(bundle.split)
    test_mask = split_arr == "test"
    test_features = bundle.features.loc[test_mask].reset_index(drop=True)
    test_labels = np.asarray(bundle.labels)[test_mask].astype(np.int8)

    start, end = first_mixed_window(test_labels)
    sliced = test_features.iloc[start:end].copy()
    sliced[label_col] = test_labels[start:end].astype(np.int8)

    out_path = DEMO_DIR / f"{name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out_path, index=False, lineterminator="\n", float_format="%.6g")

    n_rows = int(len(sliced))
    n_features = int(test_features.shape[1])
    attack_rate = float(test_labels[start:end].mean())
    print(
        f"   wrote {out_path.relative_to(ROOT)}  "
        f"rows={n_rows} features={n_features} attack_rate={attack_rate:.4f}"
    )
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "attack_rate": round(attack_rate, 6),
        "filename": out_path.name,
    }


def build_manifest() -> dict:
    """Build both slices and return the top-level manifest dict."""
    print(">> building demo datasets")

    hai = load_hai(seed=42)
    hai_entry = slice_dataset("hai_test_sample", hai)
    hai_entry.update(
        id="hai_test_sample",
        name="HAI 21.03 (test slice)",
        description=(
            "Contiguous slice of the HAI 21.03 test split with both attack and "
            "normal rows. Native sensor column names preserved (P1_*, P2_*, "
            "P3_*, P4_*). Use with HAI-trained artifacts."
        ),
        source="data/raw/hai/HAI 21.03/",
    )

    morris = load_morris(seed=42)
    morris_entry = slice_dataset("morris_gas_test_sample", morris)
    morris_entry.update(
        id="morris_gas_test_sample",
        name="Morris Gas Pipeline (test slice)",
        description=(
            "Contiguous slice of the Morris gas-pipeline test split with both "
            "attack and normal rows. IanArffDataset.arff schema preserved. "
            "Use with Morris-trained or HAI->Morris transfer artifacts."
        ),
        source="data/raw/morris/IanArffDataset.arff",
    )

    canonical_keys = (
        "id", "name", "description", "n_rows", "n_features",
        "attack_rate", "source", "filename",
    )
    return {
        "datasets": [
            {k: entry[k] for k in canonical_keys}
            for entry in (hai_entry, morris_entry)
        ]
    }


def main() -> None:
    manifest = build_manifest()
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DEMO_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"   wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
