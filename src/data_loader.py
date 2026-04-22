"""Dataset loaders for HAI and Morris Gas Pipeline.

Both loaders return a :class:`DatasetBundle` with a unified schema:
    timestamps, features (DataFrame), labels (0/1), attack_ids (str),
plus a `split` column marking train/val/test. Feature DataFrame never
contains the label or any metadata that would leak into a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RAW_DIR


@dataclass
class DatasetBundle:
    """Everything a model or metric needs about a dataset.

    Attributes:
        features: (T, F) DataFrame of numeric sensor/feature columns only.
        labels:   (T,)   int8 array, 0 = normal, 1 = any attack.
        attack_ids: (T,) object array naming the attack type per row
                    (``"normal"`` for benign rows).
        timestamps: (T,) optional datetime64 index; ``None`` when absent.
        split:    (T,)   str array with values in {"train", "val", "test"}.
                    Train/val contain *only* normal rows; test is mixed.
        name:     dataset label (``"hai"`` or ``"morris"``).
    """

    features: pd.DataFrame
    labels: np.ndarray
    attack_ids: np.ndarray
    split: np.ndarray
    name: str
    timestamps: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def mask(self, split: str) -> np.ndarray:
        return self.split == split

    def X(self, split: str) -> np.ndarray:
        return self.features.to_numpy(dtype=np.float32)[self.mask(split)]

    def y(self, split: str) -> np.ndarray:
        return self.labels[self.mask(split)]

    def assert_no_attack_in_train_val(self) -> None:
        for s in ("train", "val"):
            m = self.mask(s)
            if m.any() and int(self.labels[m].sum()) > 0:
                raise AssertionError(
                    f"Attack rows leaked into '{s}' split "
                    f"(n_attacks={int(self.labels[m].sum())}). "
                    "This would invalidate every downstream ICS metric."
                )


# ---------------------------------------------------------------------------
# HAI 21.03
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
    frames = [pd.read_csv(p) for p in sorted(paths)]  # pandas auto-detects gzip
    return pd.concat(frames, ignore_index=True)


def _detect_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _load_hai_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Back-compat: return (train_df, test_df) with `label` column but no split info.

    Kept for existing notebooks; new code should call :func:`load_hai` to get a
    :class:`DatasetBundle`.
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
        raise FileNotFoundError(f"No train*/test* CSVs found in {root}.")

    train_df = _read_hai_csvs(train_csvs)
    test_df = _read_hai_csvs(test_csvs)

    # Drop timestamp columns if non-numeric.
    for name in ("train", "test"):
        df = train_df if name == "train" else test_df
        t = _detect_col(df, HAI_TIME_COLS)
        if t is not None and not pd.api.types.is_numeric_dtype(df[t]):
            df.drop(columns=[t], inplace=True)

    # Consolidate per-process attack flags into a single `label` and keep the
    # first triggered per-process flag as the attack-id string. The bare
    # `attack` column (a global aggregate) is excluded from attack-id
    # resolution so we recover the semantically meaningful P1/P2/P3 tags.
    for df in (train_df, test_df):
        all_flags = [c for c in df.columns if c.lower().startswith("attack")]
        if not all_flags:
            continue
        per_process = [c for c in all_flags if c.lower() != "attack"]
        flag_mat = df[all_flags].to_numpy()
        df["label"] = (flag_mat.sum(axis=1) > 0).astype(np.int8)
        if per_process:
            pp_mat = df[per_process].to_numpy()
            first_hit = np.argmax(pp_mat > 0, axis=1)
            attack_id = np.array(per_process)[first_hit]
            # Rows where no per-process flag fired but `attack` did: tag "attack".
            pp_any = pp_mat.sum(axis=1) > 0
            attack_id = np.where(pp_any, attack_id, "attack")
            df["attack_id"] = np.where(df["label"].to_numpy() > 0, attack_id, "normal")
        else:
            df["attack_id"] = np.where(df["label"].to_numpy() > 0, "attack", "normal")
        df.drop(columns=all_flags, inplace=True)

    return train_df, test_df


def load_hai(val_frac: float = 0.15, seed: int = 42) -> DatasetBundle:
    """Load HAI 21.03 into a :class:`DatasetBundle` with train/val/test splits."""
    train_df, test_df = _load_hai_raw()

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_df))
    n_val = int(len(train_df) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Assemble a single frame with split labels so downstream code is uniform.
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split = np.empty(len(train_df) + len(test_df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[len(train_df):] = "test"

    combined = pd.concat([train_df, test_df], ignore_index=True)
    feature_cols = [c for c in combined.columns if c not in ("label", "attack_id")]
    features = combined[feature_cols].astype(np.float32)
    labels = combined["label"].to_numpy(dtype=np.int8)
    attack_ids = combined.get("attack_id", pd.Series(["normal"] * len(combined))).to_numpy()

    bundle = DatasetBundle(
        features=features,
        labels=labels,
        attack_ids=attack_ids,
        split=split.astype(str),
        name="hai",
        timestamps=None,
        metadata={"release": HAI_RELEASE, "val_frac": val_frac, "seed": seed},
    )
    bundle.assert_no_attack_in_train_val()
    return bundle


# ---------------------------------------------------------------------------
# Morris Gas Pipeline
# ---------------------------------------------------------------------------

MORRIS_ROOT = RAW_DIR / "morris"
MORRIS_DEFAULT_FILE = "IanArffDataset.arff"


def _parse_arff_header(path: Path) -> tuple[list[str], int]:
    """Return (attribute_names, data_start_lineno) for an ARFF file."""
    attrs: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if low.startswith("@attribute"):
                rest = s.split(None, 1)[1]
                name = rest.split("'", 2)[1] if rest.startswith("'") else rest.split(None, 1)[0]
                attrs.append(name)
            elif low.startswith("@data"):
                return attrs, i
    raise ValueError(f"No @data section found in {path}")


_MORRIS_LABEL_CANDIDATES = (
    "binary result", "binary_result", "result", "class", "label",
)
_MORRIS_ATTACK_ID_CANDIDATES = (
    "categorized result", "categorized_result", "specific result", "result",
)
_MORRIS_LEAK_COLS = frozenset({
    "binary result", "binary_result",
    "categorized result", "categorized_result",
    "specific result", "specific_result",
    "result", "class", "time",
})
_MORRIS_SENTINEL_THR = 1e9


def prepare_morris_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw Morris-family DataFrame to the canonical schema.

    Detects the binary label column (name from :data:`_MORRIS_LABEL_CANDIDATES`),
    derives an ``attack_id`` column, drops leak/metadata/time columns, coerces
    the remaining columns to numeric, and masks Morris's float32-MAX "no
    reading" sentinel (values with ``|x| > 1e9``) to zero.

    The returned frame has ``label`` and ``attack_id`` as the last two
    columns; all other columns are float features ready for scaling.

    Raises :class:`KeyError` if no label column is found — callers (web app,
    CLI) can catch this and surface a clear error to the user.
    """
    df = df.copy()

    label_col = next((c for c in _MORRIS_LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise KeyError(
            f"No Morris label column found. Expected one of {list(_MORRIS_LABEL_CANDIDATES)}; "
            f"got columns: {list(df.columns)}"
        )

    attack_id_col = next((c for c in _MORRIS_ATTACK_ID_CANDIDATES if c in df.columns), None)

    col = df[label_col]
    if pd.api.types.is_numeric_dtype(col):
        df["label"] = col.fillna(0).astype(np.int8).clip(0, 1)
    else:
        raw = col.astype(str).str.strip().str.lower()
        df["label"] = (~raw.str.startswith("normal")).astype(np.int8)

    if attack_id_col is not None:
        df["attack_id"] = df[attack_id_col].astype(str).where(df["label"] == 1, "normal")
    else:
        df["attack_id"] = np.where(df["label"] == 1, "attack", "normal")

    df = df.drop(columns=[c for c in df.columns if c in _MORRIS_LEAK_COLS])

    feature_cols = [c for c in df.columns if c not in ("label", "attack_id")]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    # Morris uses values near float32-MAX (~3.4e38) as a "no reading"
    # sentinel (notably in 'pressure measurement'). These would dominate
    # MinMax scaling and the schema-align mean. Treat any magnitude above
    # a sane engineering range as missing.
    df[feature_cols] = df[feature_cols].mask(
        df[feature_cols].abs() > _MORRIS_SENTINEL_THR, 0.0
    )
    return df.reset_index(drop=True)


def read_morris_arff(path: Path) -> pd.DataFrame:
    """Parse a Morris-family ARFF file into a raw DataFrame (pre-normalisation)."""
    attrs, data_line = _parse_arff_header(path)
    return pd.read_csv(
        path,
        header=None,
        names=attrs,
        skiprows=data_line,
        na_values=["?"],
        on_bad_lines="skip",
        engine="python",
    )


def _load_morris_raw(filename: str = MORRIS_DEFAULT_FILE) -> pd.DataFrame:
    """Parse Morris ARFF and return a DataFrame with `label` and `attack_id`."""
    path = MORRIS_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    return prepare_morris_frame(read_morris_arff(path))


def load_morris(
    val_frac: float = 0.15,
    test_frac: float = 0.3,
    seed: int = 42,
    filename: str = MORRIS_DEFAULT_FILE,
) -> DatasetBundle:
    """Morris → unified :class:`DatasetBundle`.

    Splits: all attack rows go to test; normal rows are partitioned
    train/val/test by ``1 - val_frac - test_frac`` / ``val_frac`` / ``test_frac``.
    """
    df = _load_morris_raw(filename=filename)

    normal_idx = np.flatnonzero(df["label"].to_numpy() == 0)
    attack_idx = np.flatnonzero(df["label"].to_numpy() == 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(normal_idx)
    n_test = int(len(normal_idx) * test_frac)
    n_val = int(len(normal_idx) * val_frac)
    test_normal = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_normal] = "test"
    split[attack_idx] = "test"

    feature_cols = [c for c in df.columns if c not in ("label", "attack_id")]
    bundle = DatasetBundle(
        features=df[feature_cols].astype(np.float32).reset_index(drop=True),
        labels=df["label"].to_numpy(dtype=np.int8),
        attack_ids=df["attack_id"].to_numpy(),
        split=split.astype(str),
        name="morris",
        timestamps=None,
        metadata={"filename": filename, "val_frac": val_frac, "test_frac": test_frac, "seed": seed},
    )
    bundle.assert_no_attack_in_train_val()
    return bundle


# ---------------------------------------------------------------------------
# Back-compat wrappers for existing notebooks (01/02 call these directly).
# ---------------------------------------------------------------------------

def load_hai_legacy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kept for notebooks 01/02; new code should use :func:`load_hai`."""
    return _load_hai_raw()


def morris_train_test_split(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    """Kept for notebooks 01/02; new code should use :func:`load_morris`."""
    rng = np.random.default_rng(seed)
    normal = df[df["label"] == 0].reset_index(drop=True)
    attack = df[df["label"] == 1].reset_index(drop=True)
    n_test_normal = int(len(normal) * test_frac)
    idx = rng.permutation(len(normal))
    test_normal_idx, train_idx = idx[:n_test_normal], idx[n_test_normal:]
    train_df = normal.iloc[train_idx].reset_index(drop=True)
    test_df = (
        pd.concat([normal.iloc[test_normal_idx], attack], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, test_df


def load_morris_legacy(filename: str = MORRIS_DEFAULT_FILE) -> pd.DataFrame:
    """Kept for notebooks 01/02; new code should use :func:`load_morris`."""
    return _load_morris_raw(filename)
