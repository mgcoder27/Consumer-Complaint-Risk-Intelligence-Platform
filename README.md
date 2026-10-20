# Consumer Complaint Risk Intelligence Platform

## Stage 1: MySQL data engineering

This stage loads a public CFPB Consumer Complaint Database CSV into a MySQL
star schema. It creates `fact_complaints` and the `dim_company`,
`dim_product_issue`, `dim_state`, and `dim_date` dimensions. The loader uses
bounded CSV chunks, adds missing dimension values, and can be rerun to update
existing complaint IDs.

### Project folders

- `data/raw/` — place the downloaded CFPB CSV here.
- `data/processed/` — reserved for later processing stages.
- `sql/` — MySQL table definitions and reporting queries.
- `notebooks/` — reserved for the later analysis and modeling stages.
- `src/` — reusable Python scripts, including the MySQL loader.

### Requirements and setup

1. Install the Python dependencies from the project directory:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Download the CFPB complaint CSV and save it as
   `data/raw/complaints.csv`. Inspect the first 1,000 rows before loading:

   ```powershell
   python src/inspect_sample.py
   ```

   Supply a different path or sample size if needed:

   ```powershell
   python src/inspect_sample.py "D:\data\cfpb_complaints.csv" --sample-size 500
   ```

   The utility prints the CSV headers, missing-value counts, and an exploratory
   monetary-relief class count derived from `Company response to consumer`.
   Blank responses are counted separately as `Unknown response`. These sample
   counts are for initial inspection; they are not a model evaluation.

3. Create a MySQL database, for example `cfpb_complaints`.
4. Create the tables by running `sql/schema.sql` against the chosen database.
   For example, from a MySQL command prompt:

   ```text
   source D:/path/to/project/sql/schema.sql
   ```

5. Set `MYSQL_URL` in the shell. Use your own local credentials; do not put
   credentials in the source code:

   ```powershell
   $env:MYSQL_URL = "mysql+mysqlconnector://user:password@localhost:3306/cfpb_complaints"
   ```

6. Run the loader:

   ```powershell
   python src/load_complaints.py
   ```

   A different CSV path or chunk size can be supplied:

   ```powershell
   python src/load_complaints.py "D:\data\cfpb_complaints.csv" --chunksize 2000
   ```

The loader interprets `Company response to consumer` as `company_response`,
and converts CFPB Yes/No response fields to `1`/`0`; unavailable values are
stored as SQL `NULL`. Blank company, product, issue, or state values are grouped
under an explicit `Unknown` dimension value. Narrative and response text are
kept nullable. Invalid complaint IDs, dates, or Yes/No values stop the load
with a clear error rather than being silently discarded.

The required fact columns `product_id` and `issue_id` reference the pair in
`dim_product_issue`. Since the requested design combines both concepts in one
dimension, these columns store the trimmed CFPB product and issue labels as
composite natural keys rather than separate numeric surrogate keys.

### Reporting queries

Run the statements in `sql/reporting_queries.sql` against the loaded database.

1. **Complaints per product per month** groups complaint counts by product and
   the received year and month, helping identify product-level volume changes.
2. **Monetary-relief rate by company** counts responses whose text contains
   “monetary relief” and divides by all complaints for that company. This is a
   descriptive rate, not a prediction of future relief.
3. **Top issues per state** counts complaints by state and issue, then uses
   `ROW_NUMBER()` to return each state's five most frequent issues.
4. **Month-over-month trend** totals complaints by month and uses `LAG()` to
   show the previous month's volume, absolute change, and percentage change.
   The first month has no previous-month comparison and therefore returns NULL
   for the change values.
5. **Top 10 companies by complaint volume** ranks companies by total complaints
   to surface the largest complaint workloads.
6. **Timely-response rate by company** reports timely and late response counts.
   The rate denominator includes only records marked Yes or No, excluding
   unknown response values.

The SQL uses MySQL 8 window functions for the per-state ranking and
month-over-month comparison.
