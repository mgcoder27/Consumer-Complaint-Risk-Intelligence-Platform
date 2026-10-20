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

## Stage 2: Regex narrative features

`src/regex_features.py` contains reusable text-cleaning and feature-extraction
functions. The notebook `notebooks/02_regex_features.ipynb` demonstrates five
synthetic before/after examples and explains each pattern. The examples are
fabricated and do not contain CFPB consumer information.

To process the downloaded CFPB CSV in chunks and save a feature-enriched copy:

```powershell
python src/regex_features.py
```

The defaults read `data/raw/complaints.csv` and write
`data/processed/complaints_regex_features.csv`. Supply different input/output
paths and a chunk size when needed:

```powershell
python src/regex_features.py "D:\data\complaints.csv" "D:\data\complaints_features.csv" --chunksize 20000
```

The extractor adds `narrative_clean`, `amount_mentioned` (the first explicit
dollar/USD amount as a number), `amount_mentions`, `n_amount_mentions`,
`has_amount`, `date_mentions`, `n_date_mentions`, `has_date`,
`has_account_mention`, `has_card_mention`, `has_account_card_mention`,
`n_masked_tokens`, and `narrative_length`. Amount and date matches require
recognizable formats; date matching does not validate calendar values or infer
ambiguous date ordering. Account/card flags indicate text mentions, not
verified exposures. Masked-token counts count visible `XXXX`, `****`, or
similar character runs and do not reconstruct hidden digits. Narrative
cleanup normalizes text but does not remove names or otherwise de-identify
consumer content. Protect processed complaint data and do not commit source
or processed personal data.

## Stage 3: Traditional NLP baseline

The notebook `notebooks/03_traditional_nlp_baseline.ipynb` compares TF-IDF +
Logistic Regression and TF-IDF + Linear SVM for the CFPB `Issue` label. Both
pipelines use `class_weight="balanced"`. The reusable implementation is in
`src/traditional_baseline.py`.

Run the baseline from the project root:

```powershell
python src/traditional_baseline.py
```

The default input is `data/raw/complaints.csv`; both the narrative and `Issue`
columns must be present. An alternate target, input file, or test split can be
provided:

```powershell
python src/traditional_baseline.py "D:\data\complaints.csv" --target-column Issue --test-size 0.2 --random-seed 42
```

Rows with blank narratives or labels are excluded. The remaining rows are
stratified into train and test sets with seed 42. The TF-IDF vectorizer is
inside each scikit-learn pipeline and is fitted only on the training split,
preventing vocabulary/IDF leakage into the held-out evaluation. The default
features use lowercase unigram and bigram TF-IDF, and both classifiers apply
balanced class weights.

The command prints macro-F1, macro precision, macro recall, macro one-vs-rest
ROC-AUC, a per-class precision/recall/F1/support report, a labeled confusion
matrix, and the five lowest-F1 classes for each model. It also writes
`traditional_baseline_report.json` and one confusion-matrix CSV per model under
`data/processed/baseline/`. The report's weakest-class list is derived from the
held-out scores; no performance scores are assumed before running on the actual
dataset. Classes with fewer than two examples cannot participate in a
stratified train/test split, so the code reports them explicitly rather than
silently dropping or merging issue labels.

## Stage 4: DistilBERT transformer comparison

`notebooks/04_transformer.ipynb` is designed for a free Google Colab GPU
runtime. Select a T4 GPU when available, mount Google Drive, and set the CFPB
CSV path in the notebook. It installs Transformers and Accelerate while using
Colab's preinstalled PyTorch, then creates a reproducible approximately
balanced sample of up to 20,000 complaints with non-empty narratives and
`Issue` labels. `src/balanced_sample.py` scans the CSV in two passes using only
those two columns, keeps a uniform random reservoir per class, and excludes
singleton classes that cannot be stratified into both train and test sets. A JSON sample manifest records
included/excluded classes, row counts, and the seed. The sample is balanced by
including the largest eligible classes that fit the row cap; inspect the
manifest because infrequent issues may be left out of this capped comparison.

The notebook reruns Stage 3's TF-IDF Logistic Regression and Linear SVM on
that same balanced sample, then passes their exact stratified 80/20 split
positions (seed 42) to DistilBERT. Thus the comparison is fair across the
three models, but its baseline scores are for the balanced sample and are not
the same as Stage 3's full-data baseline evaluation. DistilBERT is
`distilbert-base-uncased`, uses maximum sequence length 256, and trains for two
epochs with the Trainer and GPU mixed precision when available.
Training uses a fixed epoch count and does not use the held-out test set for
checkpoint selection; the test rows are used only for final prediction and
evaluation.

The comparison outputs include a macro-F1 table, per-class report, confusion
matrix, and up to five test examples in each direction where DistilBERT is
right and the best-F1 traditional model is wrong, or vice versa. These paired
examples help inspection but do not establish why a model got a case right.
Reports, the saved model, and narrative-bearing error examples are written
under `data/processed/transformer/`, which is Git-ignored. Consumer narratives
may contain personal information; only process them in Colab if permitted by
your data-handling requirements. No metrics or weakest-class claims are
hard-coded: they are generated when the notebook runs on your CFPB snapshot.
