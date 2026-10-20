"""Create a reproducible, approximately balanced complaint sample in two CSV passes."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_TARGET_COLUMN = "Issue"
DEFAULT_NARRATIVE_COLUMN = "Consumer complaint narrative"
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_RANDOM_SEED = 42
DEFAULT_CHUNK_SIZE = 100_000


def choose_balanced_classes(
    class_counts: pd.Series,
    target_size: int,
) -> tuple[list[str], int]:
    """Choose the largest equal-per-class sample not exceeding the requested size."""
    if target_size <= 0:
        raise ValueError("target_size must be greater than zero.")

    eligible_counts = class_counts[class_counts >= 2].sort_values(
        ascending=False,
        kind="mergesort",
    )
    if eligible_counts.empty:
        raise ValueError("At least one target class needs two non-empty narratives.")

    best_classes: list[str] = []
    best_per_class = 0
    best_total = 0

    # Testing each head-count finds a balanced sample with maximum usable rows.
    for class_count in range(1, len(eligible_counts) + 1):
        per_class = min(
            target_size // class_count,
            int(eligible_counts.iloc[class_count - 1]),
        )
        total_rows = class_count * per_class
        if per_class >= 2 and (
            total_rows > best_total
            or (total_rows == best_total and class_count > len(best_classes))
        ):
            best_classes = eligible_counts.index[:class_count].astype(str).tolist()
            best_per_class = per_class
            best_total = total_rows

    if not best_classes:
        raise ValueError(
            f"Could not form a stratifiable sample from target size {target_size}."
        )
    if len(best_classes) < 2:
        raise ValueError(
            "At least two target classes need enough non-empty narratives for "
            "a balanced classification sample."
        )

    return sorted(best_classes), best_per_class


def _class_rng(seed: int, class_name: str) -> Any:
    """Create a stable random generator for one class independent of Python hash salt."""
    import numpy as np

    seed_text = f"{seed}:{class_name}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def load_balanced_sample(
    input_path: Path,
    target_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    narrative_column: str = DEFAULT_NARRATIVE_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Read only needed CSV columns and perform deterministic per-class sampling."""
    if target_size <= 0:
        raise ValueError("target_size must be greater than zero.")
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")
    if not input_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {input_path.resolve()}")

    # Check source headers before scanning the input.
    source_columns = set(
        pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns
    )
    required_columns = {narrative_column, target_column}
    missing_columns = required_columns - source_columns
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    # Pass one counts eligible rows per class without retaining the narratives.
    class_counts: dict[str, int] = {}
    for chunk in pd.read_csv(
        input_path,
        usecols=[narrative_column, target_column],
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        narratives = chunk[narrative_column].fillna("").astype(str).str.strip()
        labels = chunk[target_column].fillna("").astype(str).str.strip()
        valid_labels = labels.loc[narratives.ne("") & labels.ne("")]
        chunk_counts = valid_labels.value_counts()
        for label, count in chunk_counts.items():
            class_counts[label] = class_counts.get(label, 0) + int(count)

    if not class_counts:
        raise ValueError("No rows contain both a non-empty narrative and target label.")

    selected_classes, per_class_size = choose_balanced_classes(
        pd.Series(class_counts),
        target_size,
    )
    print(
        f"Selected {len(selected_classes)} classes with up to "
        f"{per_class_size:,} rows each "
        f"(maximum sample size {len(selected_classes) * per_class_size:,})."
    )

    # Each class maintains its own random-key reservoir for chunk-size-independent
    # sampling. Keeping the smallest random keys yields a uniform sample per class.
    random_generators = {
        label: _class_rng(random_seed, label) for label in selected_classes
    }
    reservoirs: dict[str, pd.DataFrame] = {}
    source_row_id = 0
    for chunk in pd.read_csv(
        input_path,
        usecols=[narrative_column, target_column],
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        chunk["_source_row_id"] = range(source_row_id, source_row_id + len(chunk))
        source_row_id += len(chunk)

        narratives = chunk[narrative_column].fillna("").astype(str).str.strip()
        labels = chunk[target_column].fillna("").astype(str).str.strip()
        valid_rows = narratives.ne("") & labels.isin(selected_classes)
        candidates = chunk.loc[
            valid_rows,
            ["_source_row_id", narrative_column, target_column],
        ].copy()
        candidates[narrative_column] = narratives.loc[valid_rows]
        candidates[target_column] = labels.loc[valid_rows]

        # Keep only the best random keys for each class, never the full source.
        for label, class_rows in candidates.groupby(target_column, sort=False):
            reservoir = reservoirs.get(label)
            if reservoir is None:
                reservoir = class_rows.iloc[0:0].copy()

            class_rows = class_rows.copy()
            class_rows["_random_key"] = random_generators[label].random(len(class_rows))
            combined = pd.concat([reservoir, class_rows], ignore_index=True)
            reservoirs[label] = combined.nsmallest(
                per_class_size,
                columns="_random_key",
            )

    sampled_classes = [
        reservoir.drop(columns="_random_key").rename(
            columns={"_source_row_id": "source_row_id"}
        )
        for label, reservoir in sorted(reservoirs.items())
        if len(reservoir) == per_class_size
    ]
    if len(sampled_classes) != len(selected_classes):
        raise RuntimeError(
            "The second CSV pass did not produce the expected number of rows "
            "for every selected class."
        )

    # Stable sorting makes the downstream train/test split reproducible.
    sample = pd.concat(sampled_classes, ignore_index=True)
    sample = sample.sort_values(
        [target_column, "source_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    sample.insert(0, "sample_row_id", range(len(sample)))
    sample = sample[
        ["sample_row_id", "source_row_id", narrative_column, target_column]
    ]
    sample.attrs["manifest"] = {
        "requested_sample_size": target_size,
        "actual_sample_size": len(sample),
        "random_seed": random_seed,
        "target_column": target_column,
        "selected_classes": selected_classes,
        "excluded_classes": {
            str(label): count
            for label, count in class_counts.items()
            if label not in selected_classes
        },
        "rows_per_selected_class": per_class_size,
        "eligible_source_rows": sum(class_counts.values()),
    }
    return sample


def main() -> None:
    """Write a deterministic balanced sample for both baseline and transformer."""
    parser = argparse.ArgumentParser(
        description="Create a balanced CFPB narrative sample for model comparison."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Input CFPB CSV path (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="data/processed/transformer/balanced_sample.csv",
        help="Output CSV path for the reusable sample.",
    )
    parser.add_argument("--target-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    sample = load_balanced_sample(
        Path(args.input_csv),
        target_size=args.target_size,
        random_seed=args.random_seed,
        chunksize=args.chunksize,
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(sample.attrs["manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(sample):,} rows across {sample['Issue'].nunique():,} "
        f"balanced issue classes to {output_path.resolve()}."
    )


if __name__ == "__main__":
    main()
