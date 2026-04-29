"""One-time data cleaning pipeline.

Loads raw BindingDB TSV, applies all cleaning steps, and saves
a clean Parquet file ready for training.

Usage::

    python prepare_data.py
    python prepare_data.py --input datasets/BindingDB_All.tsv --output datasets/clean.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data.loader import load_bindingdb_data
from data.transform import (
    add_activity_label,
    remove_cx_notation,
    remove_duplicates,
    remove_nulls,
    tranform_ki_to_log_ki,
)

ROOT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean DTI dataset")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "datasets" / "BindingDB_All.tsv"),
        help="Path to raw BindingDB TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "datasets" / "clean.parquet"),
        help="Path to save the cleaned Parquet file",
    )
    parser.add_argument(
        "--pki-threshold",
        type=float,
        default=7.0,
        help="pKi threshold for activity label (default: 7.0)",
    )
    args = parser.parse_args()

    print(f"Loading raw data from: {args.input}")
    df = load_bindingdb_data(args.input)
    print(f"  Raw records: {df.height}")

    df = remove_cx_notation(df)
    df = remove_nulls(df)
    df = remove_duplicates(df)
    df = tranform_ki_to_log_ki(df)
    df = add_activity_label(df, pki_threshold=args.pki_threshold)

    print(f"  Clean records: {df.height}")
    print(f"  Columns: {df.columns}")
    print(df["is_active"].value_counts())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(output_path))
    print(f"\nSaved clean dataset to: {output_path}")


if __name__ == "__main__":
    main()
