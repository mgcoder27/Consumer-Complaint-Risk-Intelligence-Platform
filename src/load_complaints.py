"""Load CFPB complaint CSV rows into the MySQL star schema."""

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection


# CFPB CSV headers are used directly so the script can load the public export.
REQUIRED_COLUMNS = {
    "Complaint ID",
    "Date received",
    "Product",
    "Issue",
    "Company",
    "State",
    "Company response to consumer",
    "Timely response?",
}
OPTIONAL_COLUMNS = {"Consumer complaint narrative", "Consumer disputed?"}

CREATE_STAGING_TABLE = """
CREATE TEMPORARY TABLE complaint_load_stage (
    complaint_id BIGINT UNSIGNED NOT NULL,
    date_received DATE NOT NULL,
    company_name VARCHAR(512) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    issue_id VARCHAR(512) NOT NULL,
    state_code VARCHAR(10) NOT NULL,
    narrative TEXT NULL,
    company_response VARCHAR(255) NULL,
    timely_response TINYINT(1) NULL,
    consumer_disputed TINYINT(1) NULL
)
"""

INSERT_STAGING_ROW = text("""
INSERT INTO complaint_load_stage (
    complaint_id, date_received, company_name, product_id, issue_id, state_code,
    narrative, company_response, timely_response, consumer_disputed
) VALUES (
    :complaint_id, :date_received, :company_name, :product_id, :issue_id, :state_code,
    :narrative, :company_response, :timely_response, :consumer_disputed
)
""")


def clean_text(value: Any, default: str | None = None) -> str | None:
    """Trim CSV text and replace blank or missing values with a chosen default."""
    if pd.isna(value):
        return default

    cleaned_value = str(value).strip()
    return cleaned_value if cleaned_value else default


def parse_yes_no(value: Any, column_name: str) -> int | None:
    """Convert CFPB Yes/No values to 1/0 while keeping unavailable values NULL."""
    cleaned_value = clean_text(value)
    if cleaned_value is None or cleaned_value.casefold() in {"n/a", "na", "unknown"}:
        return None
    if cleaned_value.casefold() == "yes":
        return 1
    if cleaned_value.casefold() == "no":
        return 0

    raise ValueError(
        f"Unexpected value {cleaned_value!r} in {column_name!r}; expected Yes, No, or blank."
    )


def make_records(chunk: pd.DataFrame, csv_row_offset: int) -> list[dict[str, Any]]:
    """Validate and normalize one CSV chunk before sending it to MySQL."""
    records: list[dict[str, Any]] = []
    for row_number, row in chunk.iterrows():
        complaint_id_text = clean_text(row["Complaint ID"])
        try:
            complaint_id = int(complaint_id_text) if complaint_id_text is not None else -1
        except ValueError as error:
            raise ValueError(
                f"Invalid Complaint ID at CSV data row {csv_row_offset + row_number + 2}: "
                f"{row['Complaint ID']!r}."
            ) from error
        if complaint_id < 0:
            raise ValueError(
                f"Invalid Complaint ID at CSV data row {csv_row_offset + row_number + 2}: "
                f"{row['Complaint ID']!r}."
            )

        received_date = pd.to_datetime(row["Date received"], errors="coerce")
        if pd.isna(received_date):
            raise ValueError(
                f"Invalid Date received at CSV data row {csv_row_offset + row_number + 2}: "
                f"{row['Date received']!r}."
            )

        records.append(
            {
                "complaint_id": complaint_id,
                "date_received": received_date.date(),
                "company_name": clean_text(row["Company"], "Unknown"),
                "product_id": clean_text(row["Product"], "Unknown"),
                "issue_id": clean_text(row["Issue"], "Unknown"),
                "state_code": clean_text(row["State"], "UNKNOWN").upper(),
                "narrative": clean_text(row["Consumer complaint narrative"]),
                "company_response": clean_text(row["Company response to consumer"]),
                "timely_response": parse_yes_no(row["Timely response?"], "Timely response?"),
                "consumer_disputed": parse_yes_no(
                    row.get("Consumer disputed?"), "Consumer disputed?"
                ),
            }
        )

    return records


def insert_dimensions(connection: Connection) -> None:
    """Insert any new dimension members before creating the corresponding facts."""
    connection.execute(text("""
        INSERT IGNORE INTO dim_company (company_name)
        SELECT DISTINCT company_name FROM complaint_load_stage
    """))
    connection.execute(text("""
        INSERT IGNORE INTO dim_product_issue (product_id, issue_id)
        SELECT DISTINCT product_id, issue_id FROM complaint_load_stage
    """))
    connection.execute(text("""
        INSERT IGNORE INTO dim_state (state_code)
        SELECT DISTINCT state_code FROM complaint_load_stage
    """))
    connection.execute(text("""
        INSERT IGNORE INTO dim_date (date_id, date_value, year, month, month_name, quarter)
        SELECT DISTINCT
            CAST(DATE_FORMAT(date_received, '%Y%m%d') AS UNSIGNED),
            date_received,
            YEAR(date_received),
            MONTH(date_received),
            DATE_FORMAT(date_received, '%M'),
            QUARTER(date_received)
        FROM complaint_load_stage
    """))


def insert_facts(connection: Connection) -> None:
    """Resolve dimension keys and insert or refresh each complaint fact."""
    connection.execute(text("""
        INSERT INTO fact_complaints (
            complaint_id, date_received_id, company_id, product_id, issue_id,
            state_id, narrative, company_response, timely_response, consumer_disputed
        )
        SELECT
            stage.complaint_id,
            dates.date_id,
            companies.company_id,
            stage.product_id,
            stage.issue_id,
            states.state_id,
            stage.narrative,
            stage.company_response,
            stage.timely_response,
            stage.consumer_disputed
        FROM complaint_load_stage AS stage
        JOIN dim_date AS dates
            ON dates.date_value = stage.date_received
        JOIN dim_company AS companies
            ON companies.company_name = stage.company_name
        JOIN dim_product_issue AS products
            ON products.product_id = stage.product_id
           AND products.issue_id = stage.issue_id
        JOIN dim_state AS states
            ON states.state_code = stage.state_code
        ON DUPLICATE KEY UPDATE
            date_received_id = VALUES(date_received_id),
            company_id = VALUES(company_id),
            product_id = VALUES(product_id),
            issue_id = VALUES(issue_id),
            state_id = VALUES(state_id),
            narrative = VALUES(narrative),
            company_response = VALUES(company_response),
            timely_response = VALUES(timely_response),
            consumer_disputed = VALUES(consumer_disputed)
    """))


def load_csv(csv_path: Path, mysql_url: str, chunksize: int) -> int:
    """Read, validate, and load a CSV in bounded, transaction-safe chunks."""
    # Check the header first so missing or misspelled CFPB columns fail clearly.
    csv_columns = set(pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns)
    missing_columns = REQUIRED_COLUMNS - csv_columns
    if missing_columns:
        missing_column_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required CFPB columns: {missing_column_list}")

    engine = create_engine(mysql_url)
    processed_rows = 0

    # One connection keeps the temporary staging table available between chunks.
    try:
        with engine.connect() as connection:
            connection.execute(text(CREATE_STAGING_TABLE))
            connection.commit()

            # Loading bounded chunks avoids reading a potentially large CFPB export at once.
            csv_chunks = pd.read_csv(
                csv_path,
                chunksize=chunksize,
                encoding="utf-8-sig",
                dtype={"Complaint ID": "string"},
            )
            for chunk in csv_chunks:
                # Optional columns vary across CFPB exports, so add absent fields as NULL.
                for column in OPTIONAL_COLUMNS - set(chunk.columns):
                    chunk[column] = None

                records = make_records(chunk, processed_rows)
                if not records:
                    continue

                # Each chunk commits independently, avoiding one huge transaction.
                with connection.begin():
                    connection.execute(INSERT_STAGING_ROW, records)
                    insert_dimensions(connection)
                    insert_facts(connection)
                    connection.execute(text("DELETE FROM complaint_load_stage"))

                processed_rows += len(records)
                print(f"Processed {processed_rows:,} CSV rows.")
    finally:
        engine.dispose()
    return processed_rows


def main() -> None:
    """Parse command-line options and start the CSV load."""
    parser = argparse.ArgumentParser(
        description="Load a CFPB Consumer Complaint CSV into MySQL."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Path to the CFPB CSV (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of CSV rows to process per transaction (default: 5000).",
    )
    args = parser.parse_args()

    if args.chunksize <= 0:
        parser.error("--chunksize must be greater than zero.")

    # Keep credentials out of source files and provide a clear setup error.
    mysql_url = os.getenv("MYSQL_URL")
    if not mysql_url:
        raise RuntimeError(
            "Set MYSQL_URL, for example "
            "'mysql+mysqlconnector://user:password@localhost:3306/cfpb_complaints'."
        )

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {csv_path.resolve()}")

    total_rows = load_csv(csv_path, mysql_url, args.chunksize)
    print(f"Load complete: {total_rows:,} CSV rows processed.")


if __name__ == "__main__":
    main()
