"""Print basic quality and monetary-relief class counts for a CFPB CSV sample."""

import argparse
from pathlib import Path
import sys

import pandas as pd

# Support the documented direct script invocation from the project root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk_model import derive_relief_target


RELIEF_RESPONSE_COLUMN = "Company response to consumer"


def main() -> None:
    """Read a small CSV sample and print its shape, missing values, and labels."""
    parser = argparse.ArgumentParser(
        description="Inspect a sample of the CFPB Consumer Complaint CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Path to the CFPB CSV (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of initial CSV rows to inspect (default: 1000).",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero.")

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {csv_path.resolve()}")

    # Limit the read so the first exploration step stays quick and memory-friendly.
    sample = pd.read_csv(csv_path, nrows=args.sample_size, encoding="utf-8-sig")

    # Print the source headers so later transformations can use the exact names.
    print(f"Sample rows loaded: {len(sample):,}")
    print("\nColumns:")
    for column in sample.columns:
        print(f"- {column}")

    # Count missing cells in the sample, not in the full source dataset.
    print("\nMissing values in this sample:")
    print(sample.isna().sum().to_string())

    # Use company response text as an exploratory proxy for monetary relief.
    if RELIEF_RESPONSE_COLUMN not in sample.columns:
        raise ValueError(
            f"CSV is missing the expected class column {RELIEF_RESPONSE_COLUMN!r}."
        )

    response_labels = sample[RELIEF_RESPONSE_COLUMN].map(derive_relief_target)
    relief_classes = response_labels.map(
        lambda result: (
            "Monetary relief"
            if result[0] == 1
            else "No monetary relief"
            if result[0] == 0
            else f"Excluded: {result[1]}"
        )
    )

    # Show observed outcome classes and statuses that should not become labels.
    print("\nMonetary-relief outcome counts:")
    print(relief_classes.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
