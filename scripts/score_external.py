"""Score an external dataset against a saved model artifact.

Usage:
    python -m scripts.score_external \
        --artifact results/checkpoints/baseline_morris_isolation_forest/seed42 \
        --input    data/raw/morris/IanArffDataset.arff \
        --out      results/external/scores.parquet

The CLI is the backbone of the external-validation pipeline: reproducing a
summary-parquet row from a saved artifact boils down to running this script
and diffing the ``metrics`` output against the original row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_artifact, score_dataframe  # noqa: E402
from src.inference.adapters import (  # noqa: E402
    expected_input_kind,
    load_file,
    needs_projection,
)
from src.transfer.schema_align import load_feature_types, project_dataframe  # noqa: E402
from src.utils import set_seed  # noqa: E402


def _format_metric_line(family: str, m: dict) -> str:
    if family == "etapr":
        return (
            f"   etapr        TaP={m['tap']:.4f} TaR={m['tar']:.4f}"
            f"  F1={m['etapr_f1']:.4f}"
        )
    return (
        f"   {family:<12} P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f}"
        f"  F1={m.get('f1', 0):.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, required=True, help="saved artifact directory")
    ap.add_argument("--input", type=Path, required=True, help="ARFF or CSV file to score")
    ap.add_argument("--out", type=Path, required=True, help="output parquet file")
    ap.add_argument(
        "--device", default="cpu", help="torch device for deep models (default: cpu)"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    artifact = load_artifact(args.artifact, device=args.device)
    print(f">> artifact {args.artifact}")
    print(
        f"   model={artifact.model.name}  trained_on={artifact.trained_on}  "
        f"threshold={artifact.threshold:.6f} ({artifact.threshold_strategy})"
    )

    kind = expected_input_kind(artifact)
    project = needs_projection(artifact)
    result = load_file(
        args.input,
        kind=kind,
        expected_features=None if project else artifact.feature_columns,
    )
    print(f"   input {args.input}  kind={kind}  rows={len(result.features)}")

    features_df = result.features
    if project:
        types_yaml = load_feature_types()
        features_df = project_dataframe(
            features_df,
            feat_to_type=types_yaml[kind],
            target_types=list(artifact.feature_columns),
            aggregations=types_yaml.get("aggregations", {}),
        )
        print(f"   projected -> {list(features_df.columns)}")

    score_result = score_dataframe(artifact, features_df, labels=result.labels)
    print(
        f"   scored {len(score_result.scores)} "
        f"{'windows' if score_result.windowed else 'rows'}  "
        f"flagged={int(score_result.flags.sum())}"
    )

    if score_result.metrics is not None:
        for family, m in score_result.metrics.items():
            print(_format_metric_line(family, m))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "score": score_result.scores,
            "flag": score_result.flags.astype("int8"),
        }
    )
    if score_result.labels is not None:
        out_df["label"] = score_result.labels
    out_df.to_parquet(args.out, index=False)
    print(f"   wrote -> {args.out}")

    if score_result.metrics is not None:
        metrics_path = args.out.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(score_result.metrics, indent=2))
        print(f"   wrote -> {metrics_path}")


if __name__ == "__main__":
    main()
