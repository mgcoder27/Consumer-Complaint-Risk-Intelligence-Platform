"""Map CFPB taxonomy values to risk categories and join state population rates."""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


RISK_CATEGORIES = (
    "Fraud/Identity Theft",
    "Billing Dispute",
    "Credit Reporting Error",
    "Debt Collection Misconduct",
    "Other",
)
DEFAULT_PRODUCT_COLUMN = "Product"
DEFAULT_ISSUE_COLUMN = "Issue"
DEFAULT_SUB_ISSUE_COLUMN = "Sub-issue"
DEFAULT_STATE_COLUMN = "State"
DEFAULT_CENSUS_STATE_COLUMN = "state"
DEFAULT_CENSUS_POPULATION_COLUMN = "population"
DEFAULT_CHUNK_SIZE = 100_000

# The category rules are intentionally visible and editable for this project.
# Match the more specific higher-risk concepts before the broader categories.
ONTOLOGY_RULES: dict[str, tuple[str, ...]] = {
    "Fraud/Identity Theft": (
        "identity theft",
        "identity stolen",
        "identity fraud",
        "fraudulent",
        "fraud",
        "unauthorized",
        "not authorized",
        "did not authorize",
        "didn't authorize",
        "account takeover",
        "stolen account",
        "stolen identity",
        "stolen card",
        "stolen funds",
        "impersonation",
        "impersonated",
        "scam",
        "forged",
        "forgery",
        "counterfeit",
        "data breach",
    ),
    "Debt Collection Misconduct": (
        "debt collection",
        "debt collector",
        "collect debt",
        "attempts to collect",
        "debt not owed",
        "debt was not owed",
        "debt i do not owe",
        "debt i don't owe",
        "false statements",
        "false representation",
        "communication tactics",
        "threatened to contact",
        "illegal action",
        "written notification about debt",
        "debt validation",
        "harassing calls",
    ),
    "Credit Reporting Error": (
        "credit reporting",
        "credit report",
        "credit reporting company's investigation",
        "credit reporting company investigation",
        "incorrect information on your report",
        "incorrect information on credit report",
        "inaccurate information on credit report",
        "problem with a credit reporting",
        "improper use of my credit report",
        "unable to get my credit report",
        "credit score",
        "credit monitoring",
    ),
    "Billing Dispute": (
        "billing dispute",
        "billing disputes",
        "billing error",
        "incorrect billing",
        "incorrect charge",
        "duplicate charge",
        "charged twice",
        "charged a fee",
        "charged fees",
        "fees or interest",
        "unexpected fee",
        "unexpected fees",
        "incorrect balance",
        "payment applied incorrectly",
        "payment not applied",
        "problem with a purchase shown on your statement",
        "problem with a purchase shown on statement",
        "dispute a charge",
        "disputed charge",
        "statement error",
        "billing statement",
    ),
}

# Official state names and postal abbreviations, plus DC and U.S. territories
# commonly encountered in CFPB and Census extracts.
STATE_NAME_TO_CODE = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "american samoa": "AS", "guam": "GU", "northern mariana islands": "MP",
    "puerto rico": "PR", "u s virgin islands": "VI", "virgin islands": "VI",
}

STATE_FIPS_TO_CODE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY", "60": "AS", "66": "GU", "69": "MP",
    "72": "PR", "78": "VI",
}
VALID_STATE_CODES = set(STATE_NAME_TO_CODE.values())


def normalize_taxonomy_value(value: Any) -> str:
    """Lowercase taxonomy text and normalize punctuation and whitespace."""
    if value is None or pd.isna(value):
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    """Match a complete normalized phrase rather than an arbitrary substring."""
    return f" {normalize_taxonomy_value(phrase)} " in f" {normalized_text} "


def map_risk_category(
    product: Any,
    issue: Any,
    sub_issue: Any = None,
) -> tuple[str, str]:
    """Return the custom risk category and concise rule evidence for one row."""
    normalized_values = {
        "product": normalize_taxonomy_value(product),
        "issue": normalize_taxonomy_value(issue),
        "sub_issue": normalize_taxonomy_value(sub_issue),
    }
    combined_text = " ".join(value for value in normalized_values.values() if value)

    for category in RISK_CATEGORIES[:-1]:
        for phrase in ONTOLOGY_RULES[category]:
            if _contains_phrase(combined_text, phrase):
                return category, phrase

    # Product-level fallbacks cover CFPB rows whose issue text is generic.
    if _contains_phrase(normalized_values["product"], "debt collection"):
        return "Debt Collection Misconduct", "CFPB product: debt collection"
    if (
        _contains_phrase(normalized_values["product"], "credit reporting")
        or _contains_phrase(
            normalized_values["product"],
            "credit reporting or other personal consumer reports",
        )
    ):
        return "Credit Reporting Error", "CFPB product: credit reporting"

    return "Other", "no ontology rule matched"


def normalize_state(value: Any) -> str | None:
    """Normalize a state name, postal code, or Census FIPS code to postal code."""
    if value is None or pd.isna(value):
        return None

    raw_value = str(value).strip()
    if not raw_value:
        return None

    # CSV readers may render a numeric state code as a float-like string.
    if re.fullmatch(r"\d+\.0", raw_value):
        raw_value = raw_value[:-2]
    if raw_value.isdigit():
        return STATE_FIPS_TO_CODE.get(raw_value.zfill(2))

    normalized = normalize_taxonomy_value(raw_value)
    if normalized in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[normalized]
    postal_code = normalized.upper()
    if postal_code in VALID_STATE_CODES:
        return postal_code
    return None


def load_state_populations(
    population_csv: Path,
    state_column: str,
    population_column: str,
) -> dict[str, int]:
    """Read and validate one positive population estimate per normalized state."""
    if not population_csv.is_file():
        raise FileNotFoundError(
            f"Population CSV file not found: {population_csv.resolve()}"
        )

    population_data = pd.read_csv(
        population_csv,
        usecols=[state_column, population_column],
        dtype={state_column: str},
        encoding="utf-8-sig",
    )
    population_values = pd.to_numeric(
        population_data[population_column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    normalized_states = population_data[state_column].map(normalize_state)
    valid_state_rows = normalized_states.notna()

    invalid_population = (
        valid_state_rows
        & (population_values.isna() | population_values.le(0))
    )
    if invalid_population.any():
        bad_rows = population_data.loc[
            invalid_population, [state_column, population_column]
        ].head(5)
        raise ValueError(
            "Population values must be positive numeric values for recognized "
            f"state rows. Examples: {bad_rows.to_dict(orient='records')}"
        )

    state_population_data = pd.DataFrame(
        {
            "state_code": normalized_states.loc[valid_state_rows],
            "population": population_values.loc[valid_state_rows],
        }
    )
    duplicate_states = state_population_data["state_code"].duplicated(keep=False)
    if duplicate_states.any():
        duplicate_codes = sorted(
            state_population_data.loc[duplicate_states, "state_code"].unique()
        )
        raise ValueError(
            "Population CSV must contain one row per state after name/code "
            f"normalization. Duplicates: {', '.join(duplicate_codes)}"
        )

    return {
        row.state_code: int(row.population)
        for row in state_population_data.itertuples(index=False)
    }


def add_ontology_and_population_features(
    dataframe: pd.DataFrame,
    state_complaint_counts: dict[str, int],
    state_populations: dict[str, int],
    product_column: str = DEFAULT_PRODUCT_COLUMN,
    issue_column: str = DEFAULT_ISSUE_COLUMN,
    sub_issue_column: str = DEFAULT_SUB_ISSUE_COLUMN,
    state_column: str = DEFAULT_STATE_COLUMN,
) -> pd.DataFrame:
    """Add risk category and state-level per-capita complaint features."""
    required_columns = {product_column, issue_column, state_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Complaints are missing required columns: {missing_list}")

    if sub_issue_column in dataframe.columns:
        sub_issues = dataframe[sub_issue_column]
    else:
        sub_issues = pd.Series(None, index=dataframe.index)

    mapped_categories = [
        map_risk_category(product, issue, sub_issue)
        for product, issue, sub_issue in zip(
            dataframe[product_column],
            dataframe[issue_column],
            sub_issues,
        )
    ]
    featured_dataframe = dataframe.copy()
    featured_dataframe["risk_category"] = [
        category for category, _ in mapped_categories
    ]
    featured_dataframe["risk_category_rule"] = [
        evidence for _, evidence in mapped_categories
    ]

    normalized_states = dataframe[state_column].map(normalize_state)
    featured_dataframe["state_code_normalized"] = normalized_states
    featured_dataframe["state_complaint_count"] = normalized_states.map(
        state_complaint_counts
    ).astype("Int64")
    featured_dataframe["state_population"] = normalized_states.map(
        state_populations
    ).astype("Int64")

    # Leave rates missing when state or population coverage is unavailable.
    featured_dataframe["complaints_per_100k"] = (
        featured_dataframe["state_complaint_count"].astype("Float64")
        / featured_dataframe["state_population"].astype("Float64")
        * 100_000
    )
    return featured_dataframe


def process_complaints_csv(
    complaints_csv: Path,
    population_csv: Path,
    output_csv: Path,
    state_population_column: str = DEFAULT_CENSUS_STATE_COLUMN,
    population_column: str = DEFAULT_CENSUS_POPULATION_COLUMN,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    product_column: str = DEFAULT_PRODUCT_COLUMN,
    issue_column: str = DEFAULT_ISSUE_COLUMN,
    sub_issue_column: str = DEFAULT_SUB_ISSUE_COLUMN,
    state_column: str = DEFAULT_STATE_COLUMN,
) -> int:
    """Count states in one CSV pass, then write ontology and rate features in chunks."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")
    if not complaints_csv.is_file():
        raise FileNotFoundError(
            f"Complaints CSV file not found: {complaints_csv.resolve()}"
        )

    populations = load_state_populations(
        population_csv,
        state_population_column,
        population_column,
    )
    headers = set(
        pd.read_csv(complaints_csv, nrows=0, encoding="utf-8-sig").columns
    )
    required_columns = {product_column, issue_column, state_column}
    missing_columns = required_columns - headers
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Complaints CSV is missing required columns: {missing_list}")

    # Count complaint records for the entire file, not only narratives or a sample.
    state_complaint_counts: dict[str, int] = {}
    for chunk in pd.read_csv(
        complaints_csv,
        usecols=[state_column],
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        normalized_states = chunk[state_column].map(normalize_state).dropna()
        for state_code, count in normalized_states.value_counts().items():
            state_complaint_counts[state_code] = (
                state_complaint_counts.get(state_code, 0) + int(count)
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            complaints_csv,
            dtype=str,
            chunksize=chunksize,
            encoding="utf-8-sig",
        )
    ):
        featured_chunk = add_ontology_and_population_features(
            chunk,
            state_complaint_counts,
            populations,
            product_column=product_column,
            issue_column=issue_column,
            sub_issue_column=sub_issue_column,
            state_column=state_column,
        )
        featured_chunk.to_csv(
            output_csv,
            index=False,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            encoding="utf-8-sig" if chunk_number == 0 else "utf-8",
        )
        processed_rows += len(featured_chunk)
        print(f"Processed {processed_rows:,} complaint rows.")

    unmatched_population_states = sorted(
        set(state_complaint_counts) - set(populations)
    )
    summary = {
        "complaints_csv": str(complaints_csv),
        "population_csv": str(population_csv),
        "output_csv": str(output_csv),
        "complaint_rows_processed": processed_rows,
        "ontology_categories": list(RISK_CATEGORIES),
        "state_count_with_complaints": len(state_complaint_counts),
        "state_count_with_population": len(
            set(state_complaint_counts) & set(populations)
        ),
        "complaint_states_without_population": unmatched_population_states,
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return processed_rows


def main() -> None:
    """Parse file paths and column names for a user-provided Census population CSV."""
    parser = argparse.ArgumentParser(
        description="Map CFPB risk categories and join state population rates."
    )
    parser.add_argument(
        "complaints_csv",
        nargs="?",
        default="data/raw/complaints.csv",
        help="CFPB complaints CSV path (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "population_csv",
        nargs="?",
        default="data/raw/state_population.csv",
        help="Census state population CSV path (default: data/raw/state_population.csv).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="data/processed/complaints_ontology_population.csv",
        help="Enriched output CSV path.",
    )
    parser.add_argument(
        "--state-population-column",
        default=DEFAULT_CENSUS_STATE_COLUMN,
        help="Census state name/code column (default: state).",
    )
    parser.add_argument(
        "--population-column",
        default=DEFAULT_CENSUS_POPULATION_COLUMN,
        help="Census resident population column (default: population).",
    )
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    processed_rows = process_complaints_csv(
        Path(args.complaints_csv),
        Path(args.population_csv),
        Path(args.output_csv),
        state_population_column=args.state_population_column,
        population_column=args.population_column,
        chunksize=args.chunksize,
    )
    print(f"Wrote {processed_rows:,} enriched complaints to {args.output_csv}.")
    print(
        "Review the adjacent .summary.json file for state population coverage."
    )


if __name__ == "__main__":
    main()
