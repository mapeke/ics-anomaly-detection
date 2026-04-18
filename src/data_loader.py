"""Dataset loaders for HAI and Morris Gas Pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RAW_DIR


# ---------------------------------------------------------------------------
# HAI 23.05
# ---------------------------------------------------------------------------

HAI_ROOT = RAW_DIR / "hai"
HAI_RELEASE = "hai-21.03"  # 20.07/21.03 are plain .csv.gz; 22.04+/23.05 require LFS
HAI_LABEL_COLS = ("attack", "Attack", "label", "Label")
HAI_TIME_COLS = ("time", "Time", "timestamp", "Timestamp")


def _find_hai_dir() -> Path:
    """Locate the HAI release directory, being robust to clone layout."""
    candidates = [
        HAI_ROOT / HAI_RELEASE,
        HAI_ROOT / "hai" / HAI_RELEASE,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    matches = list(HAI_ROOT.rglob(HAI_RELEASE))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"HAI release '{HAI_RELEASE}' not found under {HAI_ROOT}. "
        "See data/README.md for clone instructions."
    )


def _read_hai_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in sorted(paths):
        # pandas auto-detects gzip via the .gz suffix.
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True)


def _detect_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_hai() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for HAI 23.05.

    Train: normal-only operational data (train*.csv).
    Test: attack-labelled data (test*.csv) with a binary `label` column.
    """
    root = _find_hai_dir()
    patterns = ("train*.csv", "train*.csv.gz", "hai-train*.csv", "hai-train*.csv.gz")
    train_csvs: list[Path] = []
    for pat in patterns:
        train_csvs.extend(root.glob(pat))
    test_patterns = ("test*.csv", "test*.csv.gz", "hai-test*.csv", "hai-test*.csv.gz")
    test_csvs: list[Path] = []
    for pat in test_patterns:
        test_csvs.extend(root.glob(pat))
    if not train_csvs or not test_csvs:
        raise FileNotFoundError(
            f"No train*/test* CSVs found in {root}. Check dataset layout."
        )

    train_df = _read_hai_csvs(train_csvs)
    test_df = _read_hai_csvs(test_csvs)

    # Normalise the label column to `label` (binary int).
    label_col = _detect_col(test_df, HAI_LABEL_COLS)
    if label_col is None:
        # HAI 23.05 sometimes uses multiple Attack_* flags; aggregate them.
        flag_cols = [c for c in test_df.columns if c.lower().startswith("attack")]
        if flag_cols:
            test_df["label"] = (test_df[flag_cols].sum(axis=1) > 0).astype(int)
        else:
            raise KeyError("Could not locate attack/label column in HAI test CSVs.")
    elif label_col != "label":
        test_df = test_df.rename(columns={label_col: "label"})
    test_df["label"] = test_df["label"].astype(int)

    # Training data is normal by construction.
    train_df = train_df.copy()
    if _detect_col(train_df, HAI_LABEL_COLS) is None:
        train_df["label"] = 0

    # Drop timestamp columns if present (keep only numeric sensor channels).
    for name in ("train", "test"):
        df = train_df if name == "train" else test_df
        t = _detect_col(df, HAI_TIME_COLS)
        if t is not None and not pd.api.types.is_numeric_dtype(df[t]):
            df.drop(columns=[t], inplace=True)

    # Consolidate HAI's per-process attack flags into a single `label` column.
    for df in (train_df, test_df):
        attack_flags = [c for c in df.columns if c.lower().startswith("attack")]
        if attack_flags:
            df["label"] = (df[attack_flags].sum(axis=1) > 0).astype(int)
            df.drop(columns=attack_flags, inplace=True)

    return train_df, test_df


# ---------------------------------------------------------------------------
# Morris Gas Pipeline
# ---------------------------------------------------------------------------

MORRIS_ROOT = RAW_DIR / "morris"
MORRIS_DEFAULT_FILE = "IanArffDataset.arff"


def _parse_arff_header(path: Path) -> tuple[list[str], int]:
    """Return (attribute_names, data_start_lineno) for an ARFF file."""
    attrs: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                # @attribute 'name' type  — grab the quoted or bare name.
                rest = s.split(None, 1)[1]
                if rest.startswith("'"):
                    name = rest.split("'", 2)[1]
                else:
                    name = rest.split(None, 1)[0]
                attrs.append(name)
            elif low.startswith("@data"):
                return attrs, i
    raise ValueError(f"No @data section found in {path}")


def load_morris(filename: str = MORRIS_DEFAULT_FILE) -> pd.DataFrame:
    """Load the Morris gas-pipeline ARFF file into a DataFrame.

    Parses the ARFF header manually, then reads the data block with pandas
    (tolerant of malformed rows and `?` missing-value markers that trip up
    strict ARFF parsers like liac-arff).
    """
    path = MORRIS_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for retrieval instructions."
        )

    attrs, data_line = _parse_arff_header(path)
    df = pd.read_csv(
        path,
        header=None,
        names=attrs,
        skiprows=data_line,
        na_values=["?"],
        on_bad_lines="skip",
        engine="python",
    )

    # Find the label-like column.
    label_col = None
    for candidate in ("binary result", "binary_result", "result", "class", "label"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise KeyError(
            "Could not locate a label column in Morris ARFF; "
            f"available columns: {list(df.columns)}"
        )

    # Label formats vary between Morris releases:
    #   - IanArffDataset: 'binary result' — numeric 0/1 (one NaN row at tail).
    #   - gas_final.arff: categorical 'result' — 'Normal' vs attack-type strings.
    col = df[label_col]
    if pd.api.types.is_numeric_dtype(col):
        df["label"] = col.fillna(0).astype(int).clip(0, 1)
    else:
        raw = col.astype(str).str.strip().str.lower()
        df["label"] = (~raw.str.startswith("normal")).astype(int)

    # Drop label/metadata columns so they never leak into features downstream.
    # Also drop raw UNIX timestamp — monotonic across the file and would let
    # a model trivially separate train (normal-only) vs test splits.
    leak_cols = {
        "binary result",
        "binary_result",
        "categorized result",
        "categorized_result",
        "specific result",
        "specific_result",
        "result",
        "class",
        "time",
    } - {"label"}
    drop = [c for c in df.columns if c in leak_cols and c != "label"]
    df = df.drop(columns=drop)

    # Coerce feature columns to numeric (ARFF may quote them).
    for col in df.columns:
        if col == "label":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Morris rows legitimately have missing values for fields that aren't
    # relevant to a given Modbus function code. Impute with 0 rather than
    # dropping; the downstream scaler treats 0 as "field not present".
    feature_cols = [c for c in df.columns if c != "label"]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df = df.reset_index(drop=True)
    return df


def morris_train_test_split(
    df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Morris into (train=normal-only, test=mixed) for unsupervised AD.

    We withhold a random fraction of normal rows for the test set and combine
    them with all attack rows, matching the standard one-class protocol.
    """
    rng = np.random.default_rng(seed)
    normal = df[df["label"] == 0].reset_index(drop=True)
    attack = df[df["label"] == 1].reset_index(drop=True)

    n_test_normal = int(len(normal) * test_frac)
    idx = rng.permutation(len(normal))
    test_normal_idx = idx[:n_test_normal]
    train_idx = idx[n_test_normal:]

    train_df = normal.iloc[train_idx].reset_index(drop=True)
    test_df = pd.concat(
        [normal.iloc[test_normal_idx], attack], ignore_index=True
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, test_df
