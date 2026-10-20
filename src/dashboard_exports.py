"""Create aggregated Power BI CSV exports from a CFPB complaints file."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Support both direct script execution and package imports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ontology_features import (
    DEFAULT_CENSUS_POPULATION_COLUMN,
    DEFAULT_CENSUS_STATE_COLUMN,
    load_state_populations,
    map_risk_category,
    normalize_state,
)
from src.risk_model import _derive_relief_targets


DEFAULT_INPUT = Path("data/raw/complaints.csv.gz")
DEFAULT_OUTPUT_DIR = Path("data/processed/powerbi")
DEFAULT_CHUNK_SIZE = 100_000
RESPONSE_COLUMN = "Company response to consumer"
REQUIRED_COLUMNS = {
    "Date received",
    "Product",
    "Issue",
    "Company",
    "State",
    RESPONSE_COLUMN,
}


def _add_grouped_counts(
    destination: dict[str, dict[str, int]],
    group_values: pd.Series,
    frame: pd.DataFrame,
) -> None:
    """Accumulate complaint and observed-outcome counts for one dashboard group."""
    grouped = frame.groupby(group_values.to_numpy(), dropna=False, sort=False).agg(
        complaint_count=("_row_count", "sum"),
        closed_outcome_count=("_closed_outcome", "sum"),
        monetary_relief_count=("_monetary_relief", "sum"),
    )
    for group_name, row in grouped.iterrows():
        key = str(group_name) if pd.notna(group_name) else "Unknown"
        totals = destination.setdefault(
            key,
            {
                "complaint_count": 0,
                "closed_outcome_count": 0,
                "monetary_relief_count": 0,
            },
        )
        totals["complaint_count"] += int(row["complaint_count"])
        totals["closed_outcome_count"] += int(row["closed_outcome_count"])
        totals["monetary_relief_count"] += int(row["monetary_relief_count"])


def _summary_frame(
    counts: dict[str, dict[str, int]],
    group_column: str,
) -> pd.DataFrame:
    """Turn accumulated counts into a sorted table with a safe relief rate."""
    result = pd.DataFrame(
        [
            {group_column: group_name, **values}
            for group_name, values in counts.items()
        ]
    )
    if result.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "complaint_count",
                "closed_outcome_count",
                "monetary_relief_count",
                "monetary_relief_rate_pct",
            ]
        )

    result["monetary_relief_rate_pct"] = (
        100
        * result["monetary_relief_count"]
        / result["closed_outcome_count"].where(
            result["closed_outcome_count"].ne(0)
        )
    )
    return result.sort_values(
        ["complaint_count", group_column],
        ascending=[False, True],
    ).reset_index(drop=True)


def _make_monthly_frame(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Fill missing calendar months and calculate month-over-month changes."""
    monthly = _summary_frame(counts, "month")
    if monthly.empty:
        return monthly

    month_index = pd.date_range(
        start=f"{monthly['month'].min()}-01",
        end=f"{monthly['month'].max()}-01",
        freq="MS",
    ).strftime("%Y-%m")
    monthly = monthly.set_index("month").reindex(month_index, fill_value=0)
    monthly.index.name = "month"
    monthly = monthly.reset_index()
    monthly["previous_month_count"] = monthly["complaint_count"].shift(1)
    monthly["month_over_month_change"] = (
        monthly["complaint_count"] - monthly["previous_month_count"]
    )
    monthly["month_over_month_change_pct"] = (
        100
        * monthly["month_over_month_change"]
        / monthly["previous_month_count"].where(
            monthly["previous_month_count"].ne(0)
        )
    )
    return monthly


def _recommendations(
    product_counts: dict[str, dict[str, int]],
    category_counts: dict[str, dict[str, int]],
    company_counts: dict[str, dict[str, int]],
    state_frame: pd.DataFrame,
) -> list[dict[str, str]]:
    """Create four actionable recommendations tied to observed dashboard values."""
    products = _summary_frame(product_counts, "product")
    categories = _summary_frame(category_counts, "risk_category")
    companies = _summary_frame(company_counts, "company")
    recommendations: list[dict[str, str]] = []

    if not products.empty:
        top_product = products.iloc[0]
        recommendations.append(
            {
                "priority": "1",
                "recommendation": "Prioritize controls for the highest-volume product.",
                "evidence": (
                    f"{top_product['product']} had "
                    f"{int(top_product['complaint_count']):,} complaints."
                ),
                "action": (
                    "Review the product's complaint drivers, remediation queue, "
                    "and control testing coverage."
                ),
            }
        )

    if not categories.empty:
        top_category = categories.iloc[0]
        recommendations.append(
            {
                "priority": "2",
                "recommendation": "Assign focused review to the leading mapped risk category.",
                "evidence": (
                    f"{top_category['risk_category']} mapped to "
                    f"{int(top_category['complaint_count']):,} complaints."
                ),
                "action": (
                    "Sample cases in this heuristic category and validate the "
                    "mapping before treating it as confirmed misconduct."
                ),
            }
        )

    sufficiently_observed = companies.loc[
        companies["closed_outcome_count"] >= 100
    ].sort_values(
        ["monetary_relief_rate_pct", "closed_outcome_count", "company"],
        ascending=[False, False, True],
    )
    if sufficiently_observed.empty and not companies.empty:
        sufficiently_observed = companies.sort_values(
            ["monetary_relief_rate_pct", "closed_outcome_count", "company"],
            ascending=[False, False, True],
            na_position="last",
        ).head(1)
    if not sufficiently_observed.empty:
        top_company = sufficiently_observed.iloc[0]
        small_denominator_note = (
            " The rate has fewer than 100 closed outcomes; interpret cautiously."
            if int(top_company["closed_outcome_count"]) < 100
            else ""
        )
        recommendations.append(
            {
                "priority": "3",
                "recommendation": "Review the company with the highest observed relief rate.",
                "evidence": (
                    f"{top_company['company']} had a "
                    f"{top_company['monetary_relief_rate_pct']:.2f}% monetary-relief "
                    f"rate across {int(top_company['closed_outcome_count']):,} "
                    f"closed outcomes.{small_denominator_note}"
                ),
                "action": (
                    "Review the underlying cases and operational causes; this "
                    "historical CFPB disposition rate is not a bank-cost estimate."
                ),
            }
        )

    populated_states = state_frame.dropna(subset=["complaints_per_100k"])
    if not populated_states.empty:
        top_state = populated_states.sort_values(
            ["complaints_per_100k", "complaint_count", "state"],
            ascending=[False, False, True],
        ).iloc[0]
        state_evidence = (
            f"{top_state['state']} had {top_state['complaints_per_100k']:.2f} "
            "complaints per 100,000 residents."
        )
    elif not state_frame.empty:
        top_state = state_frame.sort_values(
            ["complaint_count", "state"],
            ascending=[False, True],
        ).iloc[0]
        state_evidence = (
            f"{top_state['state']} had "
            f"{int(top_state['complaint_count']):,} complaints; per-capita "
            "comparison is unavailable without Census population data."
        )
    else:
        state_evidence = "No state-level complaint counts were available."
    recommendations.append(
        {
            "priority": "4",
            "recommendation": "Use state-level complaint intensity to guide regional review.",
            "evidence": state_evidence,
            "action": (
                "Compare complaint rates with customer exposure and local "
                "reporting patterns before reallocating compliance resources."
            ),
        }
    )
    return recommendations


def export_dashboard_csvs(
    input_csv: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    population_csv: Path | None = None,
    population_state_column: str = DEFAULT_CENSUS_STATE_COLUMN,
    population_column: str = DEFAULT_CENSUS_POPULATION_COLUMN,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Aggregate a CFPB CSV, including gzip input, into Power BI-ready files."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")
    if not input_csv.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {input_csv.resolve()}")

    headers = set(
        pd.read_csv(input_csv, nrows=0, encoding="utf-8-sig").columns
    )
    missing_columns = REQUIRED_COLUMNS - headers
    if missing_columns:
        raise ValueError(
            "CFPB CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    use_columns = [
        "Date received",
        "Product",
        "Issue",
        "Company",
        "State",
        RESPONSE_COLUMN,
    ]
    if "Sub-issue" in headers:
        use_columns.append("Sub-issue")

    populations = (
        load_state_populations(
            population_csv,
            population_state_column,
            population_column,
        )
        if population_csv is not None
        else {}
    )
    monthly_counts: dict[str, dict[str, int]] = {}
    category_counts: dict[str, dict[str, int]] = {}
    product_counts: dict[str, dict[str, int]] = {}
    issue_counts: dict[str, dict[str, int]] = {}
    company_counts: dict[str, dict[str, int]] = {}
    state_counts: dict[str, dict[str, int]] = {}
    mapped_taxonomy_combinations: set[tuple[str, str, str]] = set()
    source_rows = 0

    # Read only dashboard fields in bounded chunks; narratives and identifiers
    # are deliberately omitted from every exported file.
    for chunk in pd.read_csv(
        input_csv,
        usecols=use_columns,
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        source_rows += len(chunk)
        dates = pd.to_datetime(chunk["Date received"], errors="coerce")
        if dates.isna().any():
            bad_values = chunk.loc[dates.isna(), "Date received"].head(5).tolist()
            raise ValueError(f"Invalid Date received values found: {bad_values}")

        targets, _ = _derive_relief_targets(chunk[RESPONSE_COLUMN])
        frame = pd.DataFrame(
            {
                "_row_count": 1,
                "_closed_outcome": targets.notna().astype("int64"),
                "_monetary_relief": targets.eq(1).fillna(False).astype("int64"),
            },
            index=chunk.index,
        )
        months = dates.dt.strftime("%Y-%m")
        _add_grouped_counts(monthly_counts, months, frame)

        sub_issues = (
            chunk["Sub-issue"]
            if "Sub-issue" in chunk.columns
            else pd.Series("", index=chunk.index)
        )
        taxonomy_columns = ["Product", "Issue", "Sub-issue"]
        taxonomy_rows = pd.DataFrame(
            {
                "Product": chunk["Product"],
                "Issue": chunk["Issue"],
                "Sub-issue": sub_issues,
            },
            index=chunk.index,
        )
        taxonomy_rows["Product"] = taxonomy_rows["Product"].fillna("")
        taxonomy_rows["Issue"] = taxonomy_rows["Issue"].fillna("")
        taxonomy_rows["Sub-issue"] = sub_issues.fillna("")
        unique_taxonomy = taxonomy_rows.drop_duplicates()
        unique_taxonomy["risk_category"] = [
            map_risk_category(product, issue, sub_issue)[0]
            for product, issue, sub_issue in unique_taxonomy.itertuples(
                index=False,
                name=None,
            )
        ]
        mapped_taxonomy_combinations.update(
            unique_taxonomy[taxonomy_columns].itertuples(index=False, name=None)
        )
        categorized_rows = taxonomy_rows.merge(
            unique_taxonomy,
            on=taxonomy_columns,
            how="left",
            sort=False,
        )

        _add_grouped_counts(
            category_counts,
            categorized_rows["risk_category"],
            frame,
        )
        _add_grouped_counts(product_counts, chunk["Product"].fillna("Unknown"), frame)
        _add_grouped_counts(issue_counts, chunk["Issue"].fillna("Unknown"), frame)
        company_names = (
            chunk["Company"]
            .fillna("Unknown")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip()
        )
        _add_grouped_counts(company_counts, company_names, frame)
        normalized_states = chunk["State"].map(normalize_state).fillna("Unknown")
        _add_grouped_counts(state_counts, normalized_states, frame)

    monthly_frame = _make_monthly_frame(monthly_counts)
    category_frame = _summary_frame(category_counts, "risk_category")
    category_frame["complaint_volume_rank"] = (
        category_frame["complaint_count"].rank(method="min", ascending=False).astype(int)
    )
    product_frame = _summary_frame(product_counts, "product")
    product_frame["complaint_volume_rank"] = (
        product_frame["complaint_count"].rank(method="min", ascending=False).astype(int)
    )
    issue_frame = _summary_frame(issue_counts, "issue")
    issue_frame["complaint_volume_rank"] = (
        issue_frame["complaint_count"].rank(method="min", ascending=False).astype(int)
    )
    company_frame = _summary_frame(company_counts, "company")
    company_frame["complaint_volume_rank"] = (
        company_frame["complaint_count"].rank(method="min", ascending=False).astype(int)
    )
    company_frame["relief_rate_rank_min_100_outcomes"] = pd.NA
    qualifying_companies = company_frame["closed_outcome_count"] >= 100
    company_frame.loc[
        qualifying_companies,
        "relief_rate_rank_min_100_outcomes",
    ] = (
        company_frame.loc[qualifying_companies, "monetary_relief_rate_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    state_frame = _summary_frame(state_counts, "state")
    state_frame["population"] = state_frame["state"].map(populations).astype("Int64")
    state_frame["complaints_per_100k"] = (
        100_000
        * state_frame["complaint_count"]
        / state_frame["population"].where(state_frame["population"].notna())
    )
    state_frame["complaint_rate_rank"] = pd.NA
    states_with_population = state_frame["complaints_per_100k"].notna()
    state_frame.loc[states_with_population, "complaint_rate_rank"] = (
        state_frame.loc[states_with_population, "complaints_per_100k"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    state_frame["relief_rate_rank_min_100_outcomes"] = pd.NA
    qualifying_states = state_frame["closed_outcome_count"] >= 100
    state_frame.loc[qualifying_states, "relief_rate_rank_min_100_outcomes"] = (
        state_frame.loc[qualifying_states, "monetary_relief_rate_pct"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "monthly_volume_trend.csv": monthly_frame,
        "product_summary.csv": product_frame,
        "issue_summary.csv": issue_frame,
        "risk_category_summary.csv": category_frame,
        "company_relief_rates.csv": company_frame,
        "state_complaints_per_capita.csv": state_frame,
    }
    for filename, dataframe in exports.items():
        dataframe.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
            na_rep="",
        )

    recommendations = _recommendations(
        product_counts,
        category_counts,
        company_counts,
        state_frame,
    )
    pd.DataFrame(recommendations).to_csv(
        output_dir / "business_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_driver_status = (
        "No trained model-driver artifact supplied. Train Stage 6 and add "
        "validated feature-driver output before presenting model explanations."
    )
    pd.DataFrame(
        [
            {
                "driver_rank": "",
                "model": "",
                "feature": "",
                "coefficient": "",
                "direction": "",
                "status": model_driver_status,
            }
        ]
    ).to_csv(
        output_dir / "model_drivers.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "source_file": str(input_csv),
        "source_rows": source_rows,
        "population_data_available": population_csv is not None,
        "mapped_taxonomy_combinations": len(mapped_taxonomy_combinations),
        "outputs": sorted(
            [*exports, "business_recommendations.csv", "model_drivers.csv"]
        ),
        "important_caveats": [
            "Monetary relief means CFPB's explicit Closed with monetary relief response.",
            "Relief rates use only complaints with an explicit, classifiable closed outcome.",
            "The ontology is heuristic and is not a confirmed fraud or misconduct label.",
            "Complaint-per-capita values are blank until a valid Census population CSV is supplied.",
            "No trained model driver values are emitted without validated Stage 6 output.",
        ],
    }
    (output_dir / "dashboard_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Parse options and export dashboard-ready aggregate tables."""
    parser = argparse.ArgumentParser(
        description="Create aggregated CFPB CSV exports for a Power BI dashboard."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="CFPB CSV or gzip-compressed CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for aggregated Power BI CSV exports.",
    )
    parser.add_argument(
        "--population-csv",
        type=Path,
        help="Optional Census state population CSV for per-capita complaint rates.",
    )
    parser.add_argument(
        "--population-state-column",
        default=DEFAULT_CENSUS_STATE_COLUMN,
    )
    parser.add_argument(
        "--population-column",
        default=DEFAULT_CENSUS_POPULATION_COLUMN,
    )
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    summary = export_dashboard_csvs(
        Path(args.input_csv),
        Path(args.output_dir),
        population_csv=args.population_csv,
        population_state_column=args.population_state_column,
        population_column=args.population_column,
        chunksize=args.chunksize,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Power BI exports written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
