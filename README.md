# Consumer Complaint Risk Intelligence Platform

## Business problem and project status

Compliance teams need to triage large volumes of consumer complaints, identify
products and companies that merit investigation, and monitor complaint outcomes
without confusing a complaint count with confirmed misconduct or a predicted
bank loss. This portfolio project uses CFPB complaint records to build a
reproducible data pipeline, transparent heuristic risk categories, and
Power BI-ready summaries.

The included data is a user-supplied snapshot of the public
[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
stored as `data/raw/complaints.csv.gz` with Git LFS. It contains **18,531,443
rows** and **15 columns**, but no `Consumer complaint narrative` field. The
missing narrative prevents an honest evaluation of the narrative-based regex,
traditional NLP, and transformer models on this snapshot. Stages 3, 4, and 6
are implemented as workflows, but were not trained on the supplied data; no
model scores or drivers are claimed.

### Architecture

```text
CFPB CSV / gzip CSV
        |
        +--> chunked validation and MySQL star-schema loader (Stage 1)
        +--> regex narrative features (Stage 2; requires narratives)
        +--> TF-IDF baselines (Stage 3; requires narratives)
        +--> DistilBERT Colab workflow (Stage 4; requires narratives)
        +--> heuristic ontology and optional Census join (Stage 5)
        +--> monetary-relief model workflow (Stage 6; not trained here)
        +--> chunked aggregated CSV exports --> Power BI (Stage 7)
        |
        +--> README and detailed project report (Stage 8)
```

`src/` contains reusable implementations, `sql/` contains the MySQL schema
and reports, `notebooks/` contains exploratory and modeling workflows,
`tests/` contains focused unit tests, and `data/processed/powerbi/` contains
the generated dashboard aggregates. Raw and intermediate complaint records
are not intended for Power BI export; the dashboard exporter omits complaint
IDs, ZIP codes, and narratives.

### Results: what has and has not been measured

| Model/workflow | Task and comparison | Result on supplied snapshot |
|---|---|---|
| TF-IDF Logistic Regression | Stage 3; CFPB Issue classification from narratives | **Not evaluated.** Narrative column absent; macro-F1, per-class scores, and ROC-AUC unavailable. |
| TF-IDF Linear SVM | Stage 3; same task and split as Logistic Regression | **Not evaluated.** Narrative column absent; no baseline scores available. |
| DistilBERT (`distilbert-base-uncased`) | Stage 4; same Issue task and sampled train/test split as the traditional models | **Not trained/evaluated.** Narrative column absent; no macro-F1 comparison or error examples available. |
| Structured-only Logistic Regression | Stage 6; historical explicit monetary-relief outcome | **Not trained/evaluated.** ROC-AUC, monetary-relief precision/recall, and model drivers unavailable in this environment. |
| Text + structured Logistic Regression | Stage 6; tests whether narratives add value over structure | **Unavailable for this snapshot.** No narrative field; a text-lift comparison cannot be made. |

These are unavailable results, not zero scores. The data profile is not model
performance. The Stage 7 Power BI exports are computed from the full supplied
snapshot and are summarized in the detailed report.

### Standards, conventions, and project-specific choices

| Area | Common method or convention used | Project-specific choice (not a universal standard) |
|---|---|---|
| Modeling evaluation | Hold out test data before fitting; fit preprocessing inside pipelines; use stratified splitting and training-only cross-validation | Seed 42, 80/20 holdout, three-fold CV, class weights, Stage 6 probability threshold 0.5 |
| Metrics | Macro-F1 and class-level metrics for imbalanced multiclass classification; ROC-AUC plus positive-class precision/recall for relief ranking | Selected metric set follows this portfolio’s requirements; it is not a CFPB or regulatory benchmark |
| Risk category | Make classification rules explicit and inspectable | The five-category phrase ontology, precedence, and category labels are custom heuristics |
| Relief target | Define labels from observed outcomes and exclude ambiguous/unresolved records | Only the exact `Closed with monetary relief` response is positive; specified closed responses are negative |
| Sampling | Use deterministic, bounded sampling for large files and preserve class proportions where intended | Stage 4's balanced cap of about 20,000 and Stage 6's 200,000-row proportional cap |
| State comparison | Normalize by population for a per-capita descriptive rate | Per 100,000 residents; this is not a customer-exposure rate, and the Census input has not been provided |
| Company/state rate rankings | Show denominators and avoid unstable tiny-sample rankings | Hide rate ranks below 100 classifiable closed outcomes; the 100-record cutoff is a project threshold |

For methodology, results from the supplied dataset, cautions, and
recommendations, see the [detailed project report](reports/project_report.docx).

## Stage 1: MySQL data engineering

This stage loads a public CFPB Consumer Complaint Database CSV into a MySQL
star schema. It creates `fact_complaints` and the `dim_company`,
`dim_product_issue`, `dim_state`, and `dim_date` dimensions. The loader uses
bounded CSV chunks, adds missing dimension values, and can be rerun to update
existing complaint IDs.

### Project folders

- `data/raw/` — contains the supplied CFPB snapshot as a gzip-compressed CSV.
- `data/processed/` — generated local results, including Power BI aggregate CSVs.
- `sql/` — MySQL table definitions and reporting queries.
- `notebooks/` — reserved for the later analysis and modeling stages.
- `src/` — reusable Python scripts, including the MySQL loader.

### Included CFPB data

The supplied CFPB export is stored at `data/raw/complaints.csv.gz` and tracked
with Git LFS. Its compressed size is about 333 MiB; it expands to 5.62 GB.
Pandas reads `.csv.gz` files directly when given the path. The snapshot has
18,531,443 rows and no complaint narrative column, so narrative-based stages
cannot run on this data.

### Requirements and setup

1. Install the Python dependencies from the project directory:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Inspect the first 1,000 rows of the included compressed CSV:

   ```powershell
   python src/inspect_sample.py data/raw/complaints.csv.gz
   ```

   Supply a different path or sample size if needed:

   ```powershell
   python src/inspect_sample.py "D:\data\cfpb_complaints.csv.gz" --sample-size 500
   ```

   The utility prints the CSV headers, missing-value counts, and monetary-relief
   outcome counts derived from `Company response to consumer`. Ambiguous and
   unresolved responses remain separate from binary labels. These sample counts
   are for initial inspection, not model evaluation.

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
   python src/load_complaints.py data/raw/complaints.csv.gz
   ```

   A different CSV path or chunk size can be supplied:

   ```powershell
   python src/load_complaints.py "D:\data\cfpb_complaints.csv.gz" --chunksize 2000
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
2. **Monetary-relief rate by company** counts CFPB's explicit `Closed with
   monetary relief` response and divides by all complaints for that company.
   It excludes `Closed with non-monetary relief`; this is a descriptive rate,
   not a prediction of future relief.
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

The extractor requires the CFPB `Consumer complaint narrative` column. The
attached CSV does not contain that field, so narrative cleanup and regex
extraction cannot run on that file. When the field is present, the extractor
adds `narrative_clean`, `amount_mentioned` (the first explicit
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
columns must be present. The attached CSV has no narrative field, so TF-IDF
issue classification requires a separate narrative-bearing CFPB export. An
alternate target, input file, or test split can be
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
The attached CSV has no complaint narrative column, so it cannot run this
DistilBERT stage; use a narrative-bearing CFPB export.

## Stage 5: Risk ontology and Census population features

`src/ontology_features.py` maps CFPB `Product`, `Issue`, and optional
`Sub-issue` values into five custom categories: `Fraud/Identity Theft`,
`Billing Dispute`, `Credit Reporting Error`, `Debt Collection Misconduct`, and
`Other`. Rule phrase lists and matching precedence are explicit in the Python
dictionary `ONTOLOGY_RULES`; each complaint also receives a
`risk_category_rule` feature describing the phrase or product fallback that
matched. The ontology is a transparent heuristic, not an official CFPB label or
a verified fraud decision. Review and tune it against the exact taxonomy values
in your downloaded file.

To enrich the complaint CSV with the Census state population file you provide,
save it as `data/raw/state_population.csv`. By default the population CSV needs
columns named `state` and `population`; pass the actual column names if they
differ. For example, for Census headers `NAME` and `POPESTIMATE2024`:

```powershell
python src/ontology_features.py "data/raw/complaints.csv" "data/raw/state_population.csv" "data/processed/complaints_ontology_population.csv" --state-population-column NAME --population-column POPESTIMATE2024
```

The script normalizes state names, postal codes, and FIPS codes, counts all
complaint rows by state in a first pass, then writes the original complaints
plus `risk_category`, `risk_category_rule`, `state_code_normalized`,
`state_complaint_count`, `state_population`, and
`complaints_per_100k`. The rate formula is
`state_complaint_count / state_population * 100000`. Unmatched or missing
population values remain null; duplicate recognized states and invalid
non-positive populations are reported as errors. A summary JSON next to the
output lists state coverage. Raw and processed CSVs remain Git-ignored.

Per-capita rates are fairer than raw counts when comparing states of different
sizes because they normalize complaint volume by residents. Use counts and
rates together: a small state's rate can be volatile, and population is not a
bank's customer count. State rates also do not adjust for product mix, access
to complaint channels, reporting behavior, or complaint severity. Use a Census
population estimate whose year is appropriate for the complaint period; the
feature does not automatically adjust for a mismatch in time periods.

## Stage 6: Monetary-relief risk model

`src/risk_model.py` compares class-balanced Logistic Regression models when
text exists:

- **Structured only:** regex-derived numeric/boolean narrative features,
  Product, State, and the custom ontology category.
- **Text + structured:** the same features plus unigram/bigram TF-IDF from the
  cleaned complaint narrative.

For the attached file, the narrative column is absent. The script therefore
uses its streaming sample to run only the structured model (Product, State, and
the mapped ontology category); it reports text/regex features as unavailable
and does not claim a text lift. The uploaded source was read-only profiled
without outputting complaint narratives: it has 18,531,443 rows and 15 columns,
with no `Consumer complaint narrative` column. Since that file contains no
narratives, Stages 2-4 cannot be evaluated from this particular CSV either.

Run it from the project root and pass the source path if the file is not in
`data/raw/complaints.csv`:

```powershell
python src/risk_model.py "D:\desktop\Downloads\complaints.csv\complaints.csv"
```

After cloning with Git LFS enabled, use the included compressed snapshot
directly:

```powershell
python src/risk_model.py "data/raw/complaints.csv.gz"
```

For large files, the command scans in chunks and makes a seeded, class-
proportional sample capped at 200,000 rows (`--max-rows` and `--chunksize` are
configurable). This retains the observed outcome prevalence rather than
artificially balancing the relief classes. The default 20% stratified holdout
uses seed 42. Hyperparameters are selected
by three-fold stratified ROC-AUC cross-validation using training rows only.
Imputation, scaling, one-hot categories, and TF-IDF are fitted inside the
cross-validation pipeline. The report contains held-out ROC-AUC,
monetary-relief precision and recall (at a 0.5 probability threshold), class
counts, selected parameters, and a confusion matrix. Results and predictions
are written under `data/processed/risk_model/`, which is Git-ignored.

The target is defined conservatively for observed closed outcomes:
`Closed with monetary relief` is positive. `Closed with non-monetary relief`,
`Closed with explanation`, and `Closed without relief` are negative. Missing,
in-progress, untimely, bare `Closed`, ambiguous legacy `Closed with relief`,
and other unrecognized responses are excluded rather than guessed.
The target source, `Company response to consumer`, is never included as a
predictor. `Timely response?` and `Consumer disputed?` are also explicitly
excluded because they are response/dispute workflow fields that may be known
only after intake. The notebook `notebooks/06_risk_model.ipynb` explains this
leakage control and the results.

The observed source response counts are: 222,483 `Closed with monetary relief`
positive rows; 17,623,042 rows across the three explicit negative categories;
and 685,918 ambiguous, unresolved, untimely, or missing outcomes excluded.
That is a 1.2467% positive rate among labeled closed outcomes. The data file
contains 18,531,443 records total. These are descriptive counts for the
provided file, not model metrics.

This is a historical closed-complaint outcome model, not a validated
prospective prediction at complaint intake. If it is used to forecast future
relief, define the prediction timestamp and verify that every predictor was
available by that time. CFPB company response is an outcome proxy and may not
represent the bank's actual monetary cost.

## Stage 7: Power BI dashboard exports and insights

Run the chunked exporter to create aggregated CSV tables without copying
narratives, complaint IDs, or ZIP codes into the dashboard exports:

```powershell
python src/dashboard_exports.py
```

The default input is `data/raw/complaints.csv.gz`; the exports are written to
`data/processed/powerbi/`. A state population CSV is optional. When available,
pass it and its column names to calculate complaints per 100,000 residents:

```powershell
python src/dashboard_exports.py `
  --population-csv data/raw/state_population.csv `
  --population-state-column NAME `
  --population-column POPESTIMATE2024
```

The script creates monthly volume/trend, product, issue, mapped risk-category,
company relief-rate, and state complaint/per-capita CSVs, plus four
number-backed business recommendations, a dashboard summary JSON, and a model
drivers CSV. State population and per-capita fields remain blank until Census
data is supplied. Relief rates use only explicit classifiable closed outcomes;
company/state rankings include the outcome denominator, and rate ranks are
provided only for groups with at least 100 such outcomes. The model drivers
export clearly indicates when no fitted Stage 6 driver artifact is available;
do not interpret ontology categories as model explanations.

### Suggested Power BI page layout

1. **Overview:** KPI cards for total complaints, explicit monetary-relief
   count and rate; monthly complaint volume and month-over-month change line
   chart; product volume bar chart. Use date and product slicers.
2. **Risk categories:** complaint count and relief-rate cards; category volume
   and relief-rate bars; a detail table of category and issue counts. Explain
   that category assignment is a heuristic mapping, not a verified fraud
   determination.
3. **Companies and states:** ranked company table/bar chart with complaint
   count, closed-outcome count, monetary-relief count, and relief rate; ranked
   state table/map with complaint count, Census population, complaints per
   100,000 residents, and relief rate. Keep denominator fields visible and
   avoid ranking tiny samples as if their rates were stable.
4. **Model drivers:** display validated Stage 6 model metrics and top positive
   and negative fitted coefficients only after a model has been trained and
   its driver export is available. Otherwise show the explicit unavailable
   status rather than substituting ontology categories or invented drivers.

For four actionable recommendations tied to this CFPB snapshot's observed
numbers, see `data/processed/powerbi/business_recommendations.csv`. These are
triage suggestions, not estimates of a bank's customer-level risk or monetary
cost. Per-capita rates are more comparable across state population sizes than
raw counts, but they are not customer-normalized rates and should be read
alongside complaint counts and reporting limitations.

## Stage 8: Documentation

This README is the quick-start and stage-by-stage reference. The detailed
report in `reports/project_report.docx` records the supplied data profile,
measured descriptive findings, model evaluation status, design choices versus
common methods, business recommendations, and limitations. It explicitly
separates computed dashboard statistics from model metrics that were not
obtained.
