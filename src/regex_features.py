"""Clean CFPB complaint narratives and extract beginner-friendly regex features."""

import argparse
import html
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any


# Match an amount only when it has an explicit dollar sign or USD currency label.
AMOUNT_PATTERN = re.compile(
    r"(?<![\w.])(?:USD\s*|\$\s*)"
    r"(?P<amount>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<cents>\d{1,2}))?"
    r"(?![\w,]|\.\d)",
    flags=re.IGNORECASE,
)

# Capture common numeric dates and month-name dates without trying to validate them.
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s+\d{4})?"
    r")\b",
    flags=re.IGNORECASE,
)

# Count each visible masked segment, for example XXXX, ***, or ####.
MASKED_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:X{2,}|[*#\u2022]{2,})(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)

# Remove simple HTML-like tags and non-printing control characters before measuring text.
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# These flags record whether the narrative mentions an account or a payment card.
ACCOUNT_MENTION_PATTERN = re.compile(r"\b(?:account|acct)\b", flags=re.IGNORECASE)
CARD_MENTION_PATTERN = re.compile(
    r"\b(?:credit\s+card|debit\s+card|card)\b",
    flags=re.IGNORECASE,
)


def clean_narrative(value: Any) -> str:
    """Convert a narrative value to readable, whitespace-normalized plain text."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""

    # Decode HTML entities before removing tags so encoded text remains readable.
    cleaned_text = html.unescape(str(value))
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = CONTROL_CHARACTER_PATTERN.sub(" ", cleaned_text)
    cleaned_text = WHITESPACE_PATTERN.sub(" ", cleaned_text)
    return cleaned_text.strip()


def extract_narrative_features(value: Any) -> dict[str, Any]:
    """Return cleaned text and regex-derived features for one narrative."""
    narrative_clean = clean_narrative(value)

    # Find all currency-marked amounts, then convert the first one for numeric analysis.
    amount_matches = [
        match.group("amount") + (f".{match.group('cents')}" if match.group("cents") else "")
        for match in AMOUNT_PATTERN.finditer(narrative_clean)
    ]
    amount_mentioned = (
        float(Decimal(amount_matches[0].replace(",", ""))) if amount_matches else None
    )

    # Keep date strings for traceability and counts for straightforward modeling.
    date_matches = [match.group(0) for match in DATE_PATTERN.finditer(narrative_clean)]

    # Count each visible masked group rather than attempting to infer the real value.
    masked_token_count = len(MASKED_TOKEN_PATTERN.findall(narrative_clean))

    # Record account and card mentions separately as well as in one combined flag.
    has_account_mention = bool(ACCOUNT_MENTION_PATTERN.search(narrative_clean))
    has_card_mention = bool(CARD_MENTION_PATTERN.search(narrative_clean))

    # Return a consistent set of simple values that can be added to a DataFrame.
    return {
        "narrative_clean": narrative_clean,
        "amount_mentioned": amount_mentioned,
        "has_amount": bool(amount_matches),
        "amount_mentions": " | ".join(amount_matches),
        "n_amount_mentions": len(amount_matches),
        "date_mentions": " | ".join(date_matches),
        "n_date_mentions": len(date_matches),
        "has_date": bool(date_matches),
        "has_account_mention": has_account_mention,
        "has_card_mention": has_card_mention,
        "has_account_card_mention": has_account_mention or has_card_mention,
        "n_masked_tokens": masked_token_count,
        "narrative_length": len(narrative_clean),
    }


def add_regex_features(
    dataframe: Any,
    narrative_column: str = "Consumer complaint narrative",
) -> Any:
    """Return a copy of a DataFrame with cleaned narrative and regex features."""
    # Import pandas here so the text-cleaning helpers remain usable on their own.
    import pandas as pd

    if narrative_column not in dataframe.columns:
        raise ValueError(f"DataFrame is missing narrative column {narrative_column!r}.")

    # Keep all original columns and append the same feature fields for every row.
    featured_dataframe = dataframe.copy()
    feature_rows = [
        extract_narrative_features(value)
        for value in featured_dataframe[narrative_column]
    ]
    feature_dataframe = pd.DataFrame(feature_rows, index=featured_dataframe.index)
    return pd.concat([featured_dataframe, feature_dataframe], axis=1)


def process_csv(
    input_path: Path,
    output_path: Path,
    chunksize: int = 50_000,
    narrative_column: str = "Consumer complaint narrative",
) -> int:
    """Process a CSV in chunks and write the resulting features to another CSV."""
    # Import pandas only for CSV/DataFrame work; chunking bounds memory use.
    import pandas as pd

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV file not found: {input_path.resolve()}")

    # Validate the selected field before starting an output file.
    csv_columns = set(pd.read_csv(input_path, nrows=0, encoding="utf-8-sig").columns)
    if narrative_column not in csv_columns:
        raise ValueError(f"CSV is missing narrative column {narrative_column!r}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_rows = 0

    # Replace a prior output, then append later chunks without repeating the header.
    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize, encoding="utf-8-sig")
    ):
        featured_chunk = add_regex_features(chunk, narrative_column)
        featured_chunk.to_csv(
            output_path,
            index=False,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            encoding="utf-8-sig" if chunk_number == 0 else "utf-8",
        )
        processed_rows += len(featured_chunk)
        print(f"Processed {processed_rows:,} complaint rows.")

    return processed_rows


def main() -> None:
    """Parse command-line arguments and create a processed feature CSV."""
    parser = argparse.ArgumentParser(
        description="Add regex-derived features to CFPB complaint narratives."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Source CFPB CSV (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="data/processed/complaints_regex_features.csv",
        help="Output CSV (default: data/processed/complaints_regex_features.csv).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="Rows processed per chunk (default: 50000).",
    )
    args = parser.parse_args()

    if args.chunksize <= 0:
        parser.error("--chunksize must be greater than zero.")

    row_count = process_csv(
        Path(args.input_csv),
        Path(args.output_csv),
        chunksize=args.chunksize,
    )
    print(f"Feature extraction complete: {row_count:,} rows written to {args.output_csv}.")


if __name__ == "__main__":
    main()
