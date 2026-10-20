"""Train and compare structured-only and narrative-text relief risk models."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Support both `python -m src.risk_model` and direct script execution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ontology_features import map_risk_category
from src.regex_features import extract_narrative_features


DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 3
DEFAULT_MAX_ROWS = 200_000
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_RESPONSE_COLUMN = "Company response to consumer"
DEFAULT_NARRATIVE_COLUMN = "Consumer complaint narrative"
DEFAULT_TARGET_COLUMN = "monetary_relief"
DEFAULT_OUTPUT_DIR = Path("data/processed/risk_model")

# These are outcome or post-submission fields, never predictors.
FORBIDDEN_FEATURE_COLUMNS = (
    "Company response to consumer",
    "company_response",
    "Timely response?",
    "timely_response",
    "Consumer disputed?",
    "consumer_disputed",
    DEFAULT_TARGET_COLUMN,
)

NUMERIC_FEATURES = (
    "amount_mentioned",
    "has_amount",
    "n_amount_mentions",
    "n_date_mentions",
    "has_date",
    "has_account_mention",
    "has_card_mention",
    "n_masked_tokens",
    "narrative_length",
)
CATEGORICAL_FEATURES = ("Product", "risk_category", "State")
TEXT_FEATURE = "narrative_clean"


def _derive_relief_targets(
    responses: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Vectorize the conservative CFPB disposition mapping for large CSV chunks."""
    normalized = (
        responses.fillna("")
        .astype(str)
        .str.casefold()
        .str.split()
        .str.join(" ")
    )
    non_monetary = normalized.str.contains(
        r"\bnon[\s-]+monetary relief\b",
        regex=True,
        na=False,
    )
    positive = normalized.eq("closed with monetary relief")
    negative = normalized.isin(
        {
            "closed with explanation",
            "closed without relief",
        }
    ) | non_monetary

    targets = pd.Series(pd.NA, index=responses.index, dtype="Int64")
    targets.loc[positive] = 1
    targets.loc[negative] = 0

    reasons = pd.Series(
        "unrecognized or unresolved response",
        index=responses.index,
        dtype="string",
    )
    reasons.loc[responses.isna() | normalized.eq("")] = "missing response"
    reasons.loc[
        normalized.str.contains(r"\b(?:in progress|pending)\b", regex=True, na=False)
    ] = "unresolved response"
    reasons.loc[normalized.eq("closed") | normalized.eq("closed with relief")] = (
        "ambiguous closed outcome"
    )
    reasons.loc[positive] = "closed with monetary relief"
    reasons.loc[non_monetary] = "closed with non-monetary relief"
    reasons.loc[
        normalized.isin({"closed with explanation", "closed without relief"})
    ] = "closed without explicit monetary relief"
    return targets, reasons


def sample_modeling_rows(
    csv_path: Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    chunksize: int = DEFAULT_CHUNK_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Make a deterministic, class-proportional sample without loading a huge CSV."""
    import numpy as np

    if max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {csv_path.resolve()}")

    headers = set(pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns)
    required_columns = {
        DEFAULT_RESPONSE_COLUMN,
        "Product",
        "Issue",
        "State",
    }
    missing_columns = required_columns - headers
    if missing_columns:
        raise ValueError(
            "CSV is missing required risk-model columns: "
            + ", ".join(sorted(missing_columns))
        )
    selected_columns = [
        column
        for column in (
            DEFAULT_RESPONSE_COLUMN,
            DEFAULT_NARRATIVE_COLUMN,
            "Product",
            "Issue",
            "Sub-issue",
            "State",
        )
        if column in headers
    ]

    # First pass obtains the true target counts for proportional sampling.
    target_counts = {0: 0, 1: 0}
    excluded_reasons: dict[str, int] = {}
    total_rows = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=[DEFAULT_RESPONSE_COLUMN],
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        target_labels, target_reasons = _derive_relief_targets(
            chunk[DEFAULT_RESPONSE_COLUMN]
        )
        for target, count in target_labels.value_counts().items():
            target_counts[int(target)] += int(count)
        for reason, count in target_reasons.loc[target_labels.isna()].value_counts().items():
            excluded_reasons[str(reason)] = int(count)
        total_rows += len(chunk)

    eligible_rows = sum(target_counts.values())
    sample_size = min(max_rows, eligible_rows)
    if sample_size < 2:
        raise ValueError("Fewer than two complaints have a usable closed outcome.")

    # Allocate sample slots in proportion to the observed outcome frequencies.
    raw_quotas = {
        label: sample_size * count / eligible_rows
        for label, count in target_counts.items()
    }
    quotas = {
        label: min(count, int(math.floor(raw_quotas[label])))
        for label, count in target_counts.items()
    }
    remaining_slots = sample_size - sum(quotas.values())
    labels_by_remainder = sorted(
        target_counts,
        key=lambda label: (-(raw_quotas[label] - math.floor(raw_quotas[label])), label),
    )
    for label in labels_by_remainder:
        if remaining_slots <= 0:
            break
        if quotas[label] < target_counts[label]:
            quotas[label] += 1
            remaining_slots -= 1

    # Random-key reservoirs yield reproducible uniform samples within each target.
    rngs = {
        label: np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(f"{random_seed}:{label}".encode("utf-8")).digest()[:8],
                "big",
            )
        )
        for label in target_counts
    }
    reservoirs: dict[int, pd.DataFrame] = {}
    source_row_id = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=selected_columns,
        dtype=str,
        chunksize=chunksize,
        encoding="utf-8-sig",
    ):
        chunk["_source_row_id"] = range(source_row_id, source_row_id + len(chunk))
        source_row_id += len(chunk)
        chunk["_target"], _ = _derive_relief_targets(
            chunk[DEFAULT_RESPONSE_COLUMN]
        )
        eligible = chunk.loc[chunk["_target"].notna()].copy()
        if eligible.empty:
            continue

        for label, class_rows in eligible.groupby("_target", sort=False):
            label = int(label)
            quota = quotas[label]
            class_rows = class_rows.copy()
            class_rows["_random_key"] = rngs[label].random(len(class_rows))
            reservoir = reservoirs.get(label)
            if reservoir is None:
                reservoir = class_rows.iloc[0:0].copy()
            reservoirs[label] = pd.concat(
                [reservoir, class_rows],
                ignore_index=True,
            ).nsmallest(quota, columns="_random_key")

    sampled = [
        reservoir.drop(columns="_random_key")
        for _, reservoir in sorted(reservoirs.items())
    ]
    sample = pd.concat(sampled, ignore_index=True)
    if len(sample) != sample_size:
        raise RuntimeError(
            f"Expected {sample_size} sampled rows, but retained {len(sample)}."
        )

    sample = sample.drop(columns=["_target"])
    sample = sample.rename(columns={"_source_row_id": "source_row_id"})
    sample.insert(0, "sample_row_id", range(len(sample)))
    manifest = {
        "source_csv": str(csv_path),
        "source_rows": total_rows,
        "eligible_closed_outcome_rows": eligible_rows,
        "source_target_counts": {
            "no_monetary_relief": target_counts[0],
            "monetary_relief": target_counts[1],
        },
        "excluded_outcome_counts": excluded_reasons,
        "sample_size": len(sample),
        "sample_class_counts": {
            "no_monetary_relief": quotas[0],
            "monetary_relief": quotas[1],
        },
        "random_seed": random_seed,
        "narrative_column_available": DEFAULT_NARRATIVE_COLUMN in headers,
        "columns_loaded": selected_columns,
    }
    sample.attrs["sampling_manifest"] = manifest
    return sample, manifest


def derive_relief_target(
    response: Any,
) -> tuple[int | None, str]:
    """Label explicit relief outcomes and exclude ambiguous or unresolved cases."""
    if response is None or pd.isna(response):
        return None, "missing response"

    response_text = str(response).strip()
    if not response_text:
        return None, "missing response"

    normalized_response = " ".join(response_text.casefold().split())
    if "in progress" in normalized_response or "pending" in normalized_response:
        return None, "unresolved response"

    # Check negated/ambiguous relief statuses before matching the positive phrase.
    if re.search(r"\bnon[\s-]+monetary relief\b", normalized_response):
        return 0, "closed with non-monetary relief"
    if normalized_response == "closed with monetary relief":
        return 1, "closed with monetary relief"

    # Only explicit closed non-relief CFPB outcomes are reliable negatives.
    if normalized_response in {
        "closed with explanation",
        "closed without relief",
    }:
        return 0, "closed without explicit monetary relief"

    # Bare/legacy relief labels do not say whether relief was monetary.
    if normalized_response in {"closed", "closed with relief"}:
        return None, "ambiguous closed outcome"

    # Unknown, untimely, and still open status values are not observed outcomes.
    return None, "unrecognized or unresolved response"


def prepare_risk_features(
    dataframe: pd.DataFrame,
    response_column: str = DEFAULT_RESPONSE_COLUMN,
    narrative_column: str = DEFAULT_NARRATIVE_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Create safe pre-outcome features and return labels for closed complaints."""
    required_columns = {
        response_column,
        "Product",
        "Issue",
        "State",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input data is missing required columns: {missing_list}")

    # Derive the target first but do not copy its source response into features.
    target_results = dataframe[response_column].map(derive_relief_target)
    targets = target_results.map(lambda result: result[0])
    target_reasons = target_results.map(lambda result: result[1])
    labeled_rows = targets.notna()
    excluded_reasons = (
        target_reasons.loc[~labeled_rows].value_counts().astype(int).to_dict()
    )

    narrative_available = narrative_column in dataframe.columns

    # Generate deterministic regex and ontology features before any model fitting.
    feature_records: list[dict[str, Any]] = []
    labeled_dataframe = dataframe.loc[labeled_rows]
    for _, row in labeled_dataframe.iterrows():
        if narrative_available:
            narrative_features = extract_narrative_features(row[narrative_column])
        else:
            narrative_features = {
                "narrative_clean": "",
                "amount_mentioned": None,
                "has_amount": False,
                "amount_mentions": "",
                "n_amount_mentions": 0,
                "date_mentions": "",
                "n_date_mentions": 0,
                "has_date": False,
                "has_account_mention": False,
                "has_card_mention": False,
                "has_account_card_mention": False,
                "n_masked_tokens": 0,
                "narrative_length": 0,
            }
        risk_category, _ = map_risk_category(
            row["Product"],
            row["Issue"],
            row.get("Sub-issue"),
        )
        feature_records.append(
            {
                **narrative_features,
                "Product": str(row["Product"]).strip()
                if pd.notna(row["Product"])
                else "Unknown",
                "risk_category": risk_category,
                "State": str(row["State"]).strip().upper()
                if pd.notna(row["State"])
                else "Unknown",
            }
        )

    features = pd.DataFrame(feature_records, index=labeled_dataframe.index)
    features = features.reset_index(drop=True)
    labels = targets.loc[labeled_rows].astype("int64").reset_index(drop=True)
    metadata = {
        "input_rows": len(dataframe),
        "narrative_column_available": narrative_available,
        "narrative_non_empty_rows": (
            int(dataframe[narrative_column].fillna("").astype(str).str.strip().ne("").sum())
            if narrative_available
            else 0
        ),
        "regex_features_available": narrative_available,
        "labeled_rows": len(labels),
        "excluded_rows": int((~labeled_rows).sum()),
        "excluded_reasons": excluded_reasons,
        "class_counts": {
            "no_monetary_relief": int((labels == 0).sum()),
            "monetary_relief": int((labels == 1).sum()),
        },
        "feature_columns": list(features.columns),
        "forbidden_feature_columns": list(FORBIDDEN_FEATURE_COLUMNS),
        "target_definition": (
            "1 only for the exact 'Closed with monetary relief' response; 0 for "
            "'Closed with non-monetary relief', 'Closed with explanation', and "
            "'Closed without relief'. Ambiguous, missing, unresolved, and "
            "unrecognized responses are excluded."
        ),
    }
    return features, labels, metadata


def validate_model_split(
    labels: pd.Series,
    test_size: float,
    cv_folds: int,
) -> None:
    """Check that holdout and training CV can preserve both outcome classes."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    class_counts = labels.value_counts()
    if len(class_counts) != 2:
        raise ValueError("The relief model requires both positive and negative labels.")
    minimum_class_count = int(class_counts.min())
    if minimum_class_count < cv_folds + 1:
        raise ValueError(
            f"The minority class has {minimum_class_count} rows. At least "
            f"{cv_folds + 1} are needed for a stratified holdout plus "
            f"{cv_folds}-fold training cross-validation."
        )

    test_rows = math.ceil(len(labels) * test_size)
    train_rows = len(labels) - test_rows
    if test_rows < 2 or train_rows < cv_folds * 2:
        raise ValueError(
            "The requested holdout leaves too few rows for two test classes "
            "and stratified cross-validation. Increase the data or adjust test_size."
        )


def make_feature_pipeline(
    include_text: bool,
    include_regex: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Any:
    """Build a sparse structured/text preprocessing and balanced classifier pipeline."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as error:
        raise RuntimeError(
            "Install project requirements before running Stage 6: "
            "python -m pip install -r requirements.txt"
        ) from error

    structured_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
    transformers: list[tuple[str, Any, Any]] = []
    if include_regex:
        transformers.append(
            ("numeric", structured_pipeline, list(NUMERIC_FEATURES))
        )
    transformers.append(
        ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES))
    )
    if include_text:
        transformers.append(
            (
                "narrative_text",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
                TEXT_FEATURE,
            )
        )
    feature_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        steps=[
            ("features", feature_transformer),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _load_modeling_tools() -> dict[str, Any]:
    """Import scikit-learn model/search tools with a clear setup error."""
    try:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import (
            GridSearchCV,
            StratifiedKFold,
            train_test_split,
        )
    except ImportError as error:
        raise RuntimeError(
            "Install project requirements before running Stage 6: "
            "python -m pip install -r requirements.txt"
        ) from error

    return {
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "GridSearchCV": GridSearchCV,
        "StratifiedKFold": StratifiedKFold,
        "train_test_split": train_test_split,
    }


def run_risk_model(
    dataframe: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    response_column: str = DEFAULT_RESPONSE_COLUMN,
    narrative_column: str = DEFAULT_NARRATIVE_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    sampling_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune structured-only and text-plus-structured models on one train split."""
    tools = _load_modeling_tools()
    features, labels, metadata = prepare_risk_features(
        dataframe,
        response_column=response_column,
        narrative_column=narrative_column,
    )
    validate_model_split(labels, test_size, cv_folds)

    # Split before any learned text, scaling, imputation, or categorical fitting.
    train_indices, test_indices = tools["train_test_split"](
        list(range(len(labels))),
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    x_train = features.iloc[train_indices].reset_index(drop=True)
    x_test = features.iloc[test_indices].reset_index(drop=True)
    y_train = labels.iloc[train_indices].reset_index(drop=True)
    y_test = labels.iloc[test_indices].reset_index(drop=True)

    cv = tools["StratifiedKFold"](
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_seed,
    )
    # CV and hyperparameter selection use only the training split.
    has_narratives = metadata["regex_features_available"] and (
        features[TEXT_FEATURE].fillna("").astype(str).str.strip().ne("").any()
    )
    model_grids = {
        "structured_only": {
            "pipeline": make_feature_pipeline(
                include_text=False,
                include_regex=has_narratives,
                random_seed=random_seed,
            ),
            "parameters": {"classifier__C": [0.1, 1.0, 10.0]},
        },
    }
    if has_narratives:
        model_grids["text_and_structured"] = {
            "pipeline": make_feature_pipeline(
                include_text=True,
                random_seed=random_seed,
            ),
            "parameters": {
                "classifier__C": [0.1, 1.0, 10.0],
                "features__narrative_text__ngram_range": [(1, 1), (1, 2)],
            },
        },

    model_results: dict[str, Any] = {}
    positive_label = 1
    for model_name, model_grid in model_grids.items():
        search = tools["GridSearchCV"](
            estimator=model_grid["pipeline"],
            param_grid=model_grid["parameters"],
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train)
        best_model = search.best_estimator_
        probabilities = best_model.predict_proba(x_test)[:, 1]
        predictions = best_model.predict(x_test)
        model_results[model_name] = {
            "best_parameters": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
            "test_roc_auc": float(
                tools["roc_auc_score"](y_test, probabilities)
            ),
            "positive_class_precision": float(
                tools["precision_score"](
                    y_test,
                    predictions,
                    pos_label=positive_label,
                    zero_division=0,
                )
            ),
            "positive_class_recall": float(
                tools["recall_score"](
                    y_test,
                    predictions,
                    pos_label=positive_label,
                    zero_division=0,
                )
            ),
            "classification_report": tools["classification_report"](
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["No monetary relief", "Monetary relief"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": tools["confusion_matrix"](
                y_test,
                predictions,
                labels=[0, 1],
            ).tolist(),
            "test_predictions": [
                {
                    "row_id": int(dataframe.index[features.index[position]]),
                    "actual_monetary_relief": int(y_test.iloc[offset]),
                    "predicted_monetary_relief": int(predictions[offset]),
                    "predicted_relief_probability": float(probabilities[offset]),
                }
                for offset, position in enumerate(test_indices)
            ],
        }

    metadata.update(
        {
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "test_size": test_size,
            "random_seed": random_seed,
            "cv_folds": cv_folds,
            "sampling_manifest": sampling_manifest,
            "cv_scoring": "ROC-AUC",
            "class_weight": "balanced",
            "positive_class": "Monetary relief",
            "positive_threshold": 0.5,
            "predictor_columns": [
                *([TEXT_FEATURE] if has_narratives else []),
                *(NUMERIC_FEATURES if has_narratives else []),
                *CATEGORICAL_FEATURES,
            ],
            "removed_from_predictors": list(FORBIDDEN_FEATURE_COLUMNS),
            "comparison_note": (
                "Both models use the same stratified holdout and training-only "
                "cross-validation. When a non-empty narrative column exists, "
                "structured-only uses regex-derived numeric/boolean features plus "
                "Product, risk_category, and State, while the full model adds TF-IDF. "
                "When narrative text is unavailable, regex and text features are "
                "omitted and only the structured model is run."
            ),
            "text_model_unavailable_reason": (
                None
                if has_narratives
                else (
                    "Input has no narrative column or all narrative values are "
                    "empty; text and regex feature value-added comparison cannot "
                    "be measured from this file."
                )
            ),
            "regex_features_available": bool(has_narratives),
        }
    )
    results: dict[str, Any] = {
        "metadata": metadata,
        "models": model_results,
    }
    write_risk_results(results, output_dir)
    return results


def write_risk_results(results: dict[str, Any], output_dir: Path) -> None:
    """Save a safe summary and held-out predictions without narrative text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_results = {
        "metadata": results["metadata"],
        "models": {
            name: {
                key: value
                for key, value in model_result.items()
                if key != "test_predictions"
            }
            for name, model_result in results["models"].items()
        },
    }
    (output_dir / "risk_model_report.json").write_text(
        json.dumps(report_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for model_name, model_result in results["models"].items():
        pd.DataFrame(model_result["test_predictions"]).to_csv(
            output_dir / f"{model_name}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )


def print_risk_results(results: dict[str, Any]) -> None:
    """Print target cohort, validation setup, and requested held-out metrics."""
    metadata = results["metadata"]
    print(
        f"Labeled closed complaints: {metadata['labeled_rows']:,}; "
        f"excluded: {metadata['excluded_rows']:,}"
    )
    print(f"Class counts: {metadata['class_counts']}")
    print(
        f"Train/test: {metadata['train_rows']:,}/{metadata['test_rows']:,}; "
        f"seed={metadata['random_seed']}; CV={metadata['cv_folds']}-fold "
        f"stratified, training-only"
    )
    print(f"Excluded outcome reasons: {metadata['excluded_reasons']}")

    for model_name, model_result in results["models"].items():
        print(f"\n{model_name.replace('_', ' ').title()}")
        print(f"Best parameters: {model_result['best_parameters']}")
        print(f"Best CV ROC-AUC: {model_result['best_cv_roc_auc']:.4f}")
        print(f"Test ROC-AUC: {model_result['test_roc_auc']:.4f}")
        print(
            "Monetary-relief precision: "
            f"{model_result['positive_class_precision']:.4f}"
        )
        print(
            "Monetary-relief recall: "
            f"{model_result['positive_class_recall']:.4f}"
        )
        print("Confusion matrix (rows=actual, columns=predicted; 0 then 1):")
        print(pd.DataFrame(model_result["confusion_matrix"]).to_string(index=False))


def main() -> None:
    """Read required CFPB fields and run risk model comparisons."""
    parser = argparse.ArgumentParser(
        description="Predict monetary relief with and without narrative TF-IDF."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/complaints.csv",
        help="CFPB CSV path (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local report directory (default: data/processed/risk_model).",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum proportional outcome-stratified rows to model (default: 200000).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per CSV scan chunk (default: 250000).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {csv_path.resolve()}")

    # Sample large CFPB files in bounded passes while preserving outcome prevalence.
    dataframe, sampling_manifest = sample_modeling_rows(
        csv_path,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        random_seed=args.random_seed,
    )
    results = run_risk_model(
        dataframe,
        output_dir=Path(args.output_dir),
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_seed=args.random_seed,
        sampling_manifest=sampling_manifest,
    )
    print_risk_results(results)
    print(f"\nReports written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
