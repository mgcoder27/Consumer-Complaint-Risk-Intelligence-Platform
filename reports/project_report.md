# Consumer Complaint Risk Intelligence Platform

## Executive summary

This portfolio project turns CFPB consumer complaint records into a
reproducible, auditable workflow for complaint-volume reporting, exploratory
risk triage, and (when suitable data are available) monetary-relief modeling.
Its intended audience is a bank compliance or consumer-protection team that
must decide where to review complaints, products, companies, and geographies.
The output is a prioritization aid: CFPB complaint dispositions are not a
verified bank loss, a finding of misconduct, or an individual consumer risk
score.

The supplied CFPB snapshot was fully aggregated for Stage 7. It has **18,531,443
rows**, spanning the CFPB export's 15 columns, and is included as a 333 MiB
gzip CSV under Git LFS. It has no complaint narrative field. Consequently,
regex narrative analysis, TF-IDF issue classification, DistilBERT fine-tuning,
and the text-versus-structured feature comparison were not evaluated on this
snapshot. The relief model implementation is present, but a model was not
trained and no model metrics or drivers are reported.

The full-data descriptive results include **222,483 explicit monetary-relief
dispositions**, equal to **1.2467%** of the **17,845,525** complaints with
classifiable closed outcomes under the project's target definition. Stage 7
exports eight aggregate CSV tables and a four-item recommendations CSV for
Power BI. Census population data were not supplied, so per-capita state
values remain blank.

## Business problem and intended use

A financial institution may receive thousands of complaints across different
products, issue types, locations, and service channels. Compliance operations
need a consistent way to:

- monitor complaint volumes and changes over time;
- identify recurring complaint themes for investigation;
- compare observed relief dispositions across companies;
- compare state complaint intensity without treating large-population states
  as inherently higher-risk; and
- develop predictive models only when inputs, labels, and evaluation data are
  adequate.

This project supports the first four activities descriptively and provides
model workflows for future use. It does not establish causality, regulatory
noncompliance, confirmed fraud, or an expected monetary cost to a particular
bank.

## Data source and profile

**Source:** the user-supplied export of the public
[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).
The repository snapshot is `data/raw/complaints.csv.gz`, tracked using Git
LFS. The original CSV is 5,617,801,879 bytes (about 5.23 GiB); the gzip file is
349,023,003 bytes (about 333 MiB). The exact source download date and source
query parameters were not retained as snapshot metadata; consumers should
record those when refreshing the data.

| Profile item | Observed value |
|---|---:|
| Complaint rows | 18,531,443 |
| Source columns | 15 |
| Complaint narrative column | Absent |
| `Company response to consumer` = `Closed with monetary relief` | 222,483 |
| Explicit classifiable negative outcomes | 17,623,042 |
| Total classifiable closed outcomes | 17,845,525 |
| Monetary-relief share of classifiable outcomes | 1.2467% |
| Excluded missing, ambiguous, unresolved, untimely, or unrecognized outcomes | 685,918 |

The 15 columns are `Date received`, `Product`, `Sub-product`, `Issue`,
`Sub-issue`, `Company public response`, `Company`, `State`, `ZIP code`, `Tags`,
`Submitted via`, `Date sent to company`, `Company response to consumer`,
`Timely response?`, and `Complaint ID`. The snapshot does not contain
`Consumer complaint narrative` or `Consumer disputed?`.

### Relief target definition

The binary Stage 6 target is deliberately narrow and is created from
`Company response to consumer`:

| Target | CFPB response values |
|---|---|
| Positive (1) | Exact `Closed with monetary relief` |
| Negative (0) | `Closed with non-monetary relief`, `Closed with explanation`, and `Closed without relief` |
| Excluded | Missing, in progress, untimely, bare `Closed`, ambiguous `Closed with relief`, and other unrecognized values |

This exact matching avoids the common substring error where “non-monetary
relief” is accidentally labeled as monetary relief. The mapping is a
project-specific operational definition, not a CFPB-endorsed predictive label.

## Architecture and implementation

```text
CFPB CSV or gzip CSV
   |
   +-- Stage 1: chunked CSV loader --> MySQL star schema --> SQL reporting
   +-- Stage 2: narrative cleanup and regex feature extraction
   +-- Stage 3: TF-IDF + Logistic Regression / Linear SVM
   +-- Stage 4: DistilBERT Trainer workflow (Colab GPU)
   +-- Stage 5: custom taxonomy mapping + optional Census population join
   +-- Stage 6: structured-only vs text-and-structured relief model
   +-- Stage 7: chunked aggregated CSVs --> Power BI
   +-- Stage 8: README and this report
```

- `src/` holds reusable Python implementations.
- `sql/schema.sql` defines the MySQL star schema; `sql/reporting_queries.sql`
  contains the reporting queries.
- `notebooks/` contains EDA and model workflows.
- `tests/` contains synthetic unit tests for regex features, sampling,
  ontology mapping, risk labels, and Power BI aggregation.
- `data/raw/complaints.csv.gz` is the supplied source snapshot.
- `data/processed/powerbi/` holds the aggregate dashboard exports. Raw and
  other processed complaint-level files are excluded from ordinary Git
  tracking.

The Power BI export does not include complaint IDs, ZIP codes, or narratives.
It aggregates in bounded chunks and reports the source row total and caveats
in `dashboard_export_summary.json`.

## Stage-by-stage status and how to run

Run commands from the project root. Install `requirements.txt` first for
Python workflows. A local MySQL server is needed for Stage 1; Stage 4 is
configured for Google Colab with a GPU runtime.

### Stage 1 — SQL data engineering

**Implemented:** MySQL fact table and company, product/issue, state, and date
dimensions; chunked idempotent loader; six reporting queries.

```powershell
python src/inspect_sample.py data/raw/complaints.csv.gz
# Create a database, then run sql/schema.sql using MySQL 8.
$env:MYSQL_URL = "mysql+mysqlconnector://USER:PASSWORD@localhost:3306/cfpb_complaints"
python src/load_complaints.py data/raw/complaints.csv.gz --chunksize 5000
```

No database load or MySQL query execution was performed as part of the
documented supplied-data analysis; these commands require user-provided local
database credentials and server access.

### Stage 2 — Regex narrative features

**Implemented; not applicable to this snapshot** because the narrative column
is absent. The notebook uses synthetic examples; they are not actual complaint
text.

```powershell
python src/regex_features.py "D:\data\narrative_complaints.csv" "data/processed/complaints_regex_features.csv"
```

Provide a narrative-bearing CFPB export to run this stage. The extractor
captures recognizable amounts, date strings, account/card mentions, masked
tokens, and narrative length. Regex matches are indicators, not verified
financial exposures. Text cleanup does not de-identify narratives.

### Stage 3 — Traditional NLP baseline

**Implemented; not evaluated** on the included snapshot. The task is CFPB
`Issue` classification from narratives using TF-IDF plus Logistic Regression
and Linear SVM. No macro-F1, per-class score, confusion matrix, or ROC-AUC is
available for this snapshot.

```powershell
python src/traditional_baseline.py "D:\data\narrative_complaints.csv" --target-column Issue --random-seed 42
```

The implementation stratifies the split before fitting and keeps TF-IDF
preprocessing inside the model pipeline. A narrative-bearing dataset and
installed scikit-learn are required.

### Stage 4 — DistilBERT transformer

**Notebook implemented; not trained or evaluated** because this snapshot has
no narratives. In Google Colab, open `notebooks/04_transformer.ipynb`, select
a GPU runtime if available, provide an approved narrative-bearing CSV, set
its path in the notebook, and run the cells in order. The notebook samples up
to about 20,000 records and evaluates the traditional models and
`distilbert-base-uncased` on the same split. No macro-F1 comparison, trained
checkpoint, or right/wrong example analysis is available for the supplied
snapshot.

### Stage 5 — Ontology and Census population features

**Ontology implemented; population join implemented; no Census file supplied.**
The mapping applies custom rules to CFPB product, issue, and sub-issue fields.
The population join accepts a state population file and creates complaints per
100,000 residents.

```powershell
python src/ontology_features.py `
  "data/raw/complaints.csv.gz" `
  "data/raw/state_population.csv" `
  "data/processed/complaints_ontology_population.csv" `
  --state-population-column NAME `
  --population-column POPESTIMATE2024
```

The included data cannot produce per-capita rates until a compatible Census
file is provided. The rate is population-normalized, not normalized by a
financial institution's customers or accounts.

### Stage 6 — Monetary-relief risk model

**Implemented; not trained on this snapshot.** The planned structured model
uses product, state, and ontology category; where narratives exist, it also
compares regex features and TF-IDF. The target mapping above excludes
ambiguous outcomes. The target-source response, `Timely response?`, and
`Consumer disputed?` are not predictors.

```powershell
python src/risk_model.py "data/raw/complaints.csv.gz"
```

Scikit-learn was unavailable in the development environment, and the full
modeling sample attempt did not complete under available memory constraints.
Therefore, there is no trained model, held-out ROC-AUC, monetary-relief
precision/recall, or coefficient/driver analysis to report. The outcome
prevalence shown above is descriptive, not a model score.

### Stage 7 — Power BI dashboard exports

**Run against the full supplied snapshot.** The aggregate files are already
generated at `data/processed/powerbi/`.

```powershell
python src/dashboard_exports.py
```

To calculate state per-capita values after obtaining a Census population CSV:

```powershell
python src/dashboard_exports.py `
  --population-csv data/raw/state_population.csv `
  --population-state-column NAME `
  --population-column POPESTIMATE2024
```

The model drivers table explicitly reports unavailable values until a
validated Stage 6 model exists. The four business recommendations are
exported to `business_recommendations.csv`.

### Stage 8 — Documentation

The README summarizes the business problem, architecture, setup, stage
commands, model status, design choices, and dashboard. This report records the
data profile, measured findings, limitations, and business recommendations.

## Measured descriptive findings

All counts below come from the full supplied snapshot via Stage 7. These are
descriptive aggregates, not model performance or evidence that a company
violated a rule.

### Product and category results

| View | Finding |
|---|---|
| Highest-volume product | Credit reporting or other personal consumer reports: **12,910,622 complaints** |
| Leading mapped category | Credit Reporting Error: **15,194,957 complaints** |
| Billing Dispute | 200,143 complaints; 196,917 classifiable closed outcomes; 53,051 explicit monetary-relief responses (**26.94%** of classifiable closed outcomes) |
| Other | 1,672,315 complaints; 140,824 explicit monetary-relief responses |
| Fraud/Identity Theft | 403,121 complaints; 17,115 explicit monetary-relief responses |
| Debt Collection Misconduct | 1,060,907 complaints; 5,090 explicit monetary-relief responses |

The category volume distribution reflects the custom ontology's exact phrase
rules and precedence. The 15,194,957 Credit Reporting Error records should
not be interpreted as confirmed reporting violations. Category counts sum to
the 18,531,443 source rows, and their monetary-relief counts sum to 222,483.

### Companies and states

- Premier Student Loan Center had the highest observed monetary-relief rate
  among companies with at least 100 classifiable closed outcomes:
  **68/130 = 52.31%**. This is a small denominator and a CFPB outcome proxy.
- Texas had the highest raw state volume in the exported state table:
  **2,571,937 complaints**.
- State population and complaints-per-100,000 values are blank because the
  project owner has not supplied the Census population CSV. No per-capita
  state ranking is claimed.
- Rate ranks are withheld for groups with fewer than 100 classifiable closed
  outcomes. The cutoff is a project-selected stability guard, not an
  externally mandated statistical standard.

## Results table: models

| Model | Target | Macro-F1 | ROC-AUC | Positive-class precision / recall | Status |
|---|---|---:|---:|---:|---|
| TF-IDF Logistic Regression | CFPB Issue | Not available | Not available | Not available | Not run; narrative absent |
| TF-IDF Linear SVM | CFPB Issue | Not available | Not available | Not available | Not run; narrative absent |
| DistilBERT | CFPB Issue | Not available | Not available | Not available | Not trained; narrative absent |
| Stage 6 structured-only Logistic Regression | Explicit monetary-relief response | Not available | Not available | Not available | Model not trained; scikit-learn unavailable and sample run constrained by memory |
| Stage 6 text + structured Logistic Regression | Explicit monetary-relief response | Not available | Not available | Not available | Not possible on this snapshot; narratives absent |

“Not available” means no evaluation was run; it does not mean zero performance.
The models answer different tasks (Issue classification versus relief
prediction), so any future comparison must use the appropriate target and
evaluation split. Stage 4's comparison must use the same balanced sample and
same split for the traditional and transformer models.

## Dashboard layout and business recommendations

Suggested Power BI pages:

1. **Overview:** complaint totals; explicit relief count/rate; monthly trend
   and month-over-month change; product volume.
2. **Risk categories:** category counts, closed-outcome denominators, relief
   counts/rates, and issue details, with a visible heuristic caveat.
3. **Companies and states:** company relief rate and denominator; state raw
   count and (once population data exist) population and per-capita rate.
   Show denominators and do not conceal small samples.
4. **Model drivers:** validated Stage 6 metrics and fitted model drivers only
   after training. Until then, show the unavailable-status record rather than
   substituting heuristic ontology matches.

Number-backed recommendations for a compliance head:

1. **Prioritize complaint operations and control testing for the highest-volume
   product.** Credit reporting or other personal consumer reports accounts
   for **12,910,622 complaints**. Break the workload down by issues and
   reporting periods before choosing a control intervention.
2. **Validate and investigate the largest mapped risk theme.** The ontology
   maps **15,194,957 complaints** to Credit Reporting Error. Review a sample
   of mapped records and the ontology's rules before treating this as a
   confirmed error population.
3. **Perform a case review of high observed company relief rates.** Premier
   Student Loan Center has **68 relief outcomes among 130 classifiable closed
   complaints (52.31%)**. Review the underlying dispositions and the small
   denominator; do not interpret the rate as predicted cost or bank exposure.
4. **Use state counts for initial workload planning, then add population
   context.** Texas has **2,571,937 complaints**, but there is no Census
   denominator in this project yet. Obtain time-appropriate state populations
   before comparing per-capita rates, and use customer/account exposure for
   institution-level decisions.

## Standards and design decisions

### Common, broadly used methods

- Stratified train/test splitting for classification tasks with imbalanced
  labels.
- Splitting before learned preprocessing, with vectorization, imputation,
  scaling, and encoding fitted within training pipelines.
- Training-only cross-validation and a held-out test set for final evaluation.
- Macro-F1/per-class metrics for imbalanced multiclass classification, and
  ROC-AUC plus positive-class precision/recall for binary relief prediction.
- Chunked CSV reads and a star schema as common engineering patterns for
  large data and reporting systems.
- Displaying outcome-rate denominators and using population-normalized rates
  when comparing geographic complaint counts.

These practices are conventional choices; their use does not itself establish
regulatory compliance or certify model validity.

### Explicitly project-specific choices

- The exact response-to-binary-target mapping and exclusion rules.
- Five ontology categories, phrase dictionaries, matching precedence, and
  product fallbacks.
- Seed 42, the Stage 4 balanced sample cap, and the Stage 6 proportional
  sample cap.
- The default probability threshold of 0.5 and the 100-outcome cutoff for
  company/state relief-rate ranking.
- Complaints per 100,000 residents as the selected geographic descriptive
  rate.
- Logistic Regression, TF-IDF, DistilBERT, model hyperparameter grids, and the
  selected feature set.

None of these project choices should be described as a CFPB standard or a
regulatory requirement. The ontology is not an official taxonomy, and model
coefficients (when eventually produced) would not be causal explanations.

## Limitations, privacy, and responsible use

1. **No narratives:** Stages 2–4 and text-based lift in Stage 6 cannot be
   evaluated on this snapshot. A separate narrative-bearing export is needed,
   and narratives may contain personal information.
2. **No trained model metrics:** no baseline, transformer, or Stage 6 score is
   available. Synthetic unit tests verify selected code behavior, not
   generalization to real complaint outcomes.
3. **Outcome proxy, not cost:** the CFPB disposition `Closed with monetary
   relief` does not encode the bank's actual cost, amount paid, or future
   relief likelihood.
4. **Historical timing:** the outcome is observed after complaint handling;
   the project is not validated as an intake-time prediction. Define a
   prediction timestamp and enforce feature availability if forecasting.
5. **Heuristic labels:** ontology categories are imperfect phrase mappings;
   issue/sub-issue coverage and prevalence need human review. Fraud is not
   adjudicated by this mapping.
6. **No Census denominator:** state per-capita fields are unavailable without
   population data. Even when added, state population is not the bank's
   customer base, and reporting access/behavior may differ by location.
7. **Snapshot and source provenance:** exact download date and request
   parameters were not recorded; complaint taxonomies, reporting periods, and
   company names can change over time.
8. **Confounding and representation:** complaint volume is affected by
   product mix, consumer access, awareness, reporting behavior, and company
   size. High counts do not establish a higher underlying violation rate.
9. **No causal claims:** the descriptive tables and any future predictive
   scores can prioritize review; they do not prove that a company or product
   caused harm or violated law.
10. **Resource constraints:** scikit-learn was missing in the development
    environment and an attempted full-data sample was constrained by memory.
    Training should occur in an environment provisioned for the selected
    sample and model.

## Validation record and next steps

- Stage 7 processed all **18,531,443** supplied records. Category complaint
  counts reconcile to the input row total; category positive counts reconcile
  to **222,483**.
- Stage 7 unit tests cover gzip input, chunk-boundary aggregation, monthly
  zero-fill, relief rate denominators, Census population joins, missing
  population behavior, and the explicit unavailable-model-driver status.
- Other existing tests cover regex extraction, ontology mapping, sampling,
  and relief-label leakage rules. Tests use synthetic data where the actual
  snapshot lacks required narrative or Census fields.
- Model-training, MySQL integration, and Power BI Desktop rendering were not
  validated in this environment.

Recommended follow-up sequence:

1. Record the CFPB export date and query parameters, and obtain a narrative
   export only if privacy and permitted-use requirements allow it.
2. Obtain Census state population estimates that match the complaint period,
   then regenerate the state per-capita export.
3. Install dependencies in a suitable environment and run Stage 3/4 on a
   narrative sample with the documented shared split.
4. Train Stage 6 with a feasible sample, inspect class-specific metrics and
   calibration, and validate features against the prediction timestamp.
5. Review ontology mappings with compliance subject-matter experts and
   evaluate rates with appropriate denominators before operational use.
6. Build the Power BI report from the generated aggregate CSVs and validate
   filters, sorting, and interpretation with stakeholders.
