"""Train and evaluate TF-IDF issue-classification baselines."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DEFAULT_RANDOM_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_NARRATIVE_COLUMN = "Consumer complaint narrative"
DEFAULT_TARGET_COLUMN = "Issue"


def prepare_labeled_data(
    dataframe: pd.DataFrame,
    narrative_column: str = DEFAULT_NARRATIVE_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.Series, pd.Series, int]:
    """Keep rows with both a usable narrative and a known target label."""
    missing_columns = {narrative_column, target_column} - set(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing_list}")

    # Normalize narrative text and labels before excluding unusable examples.
    narratives = dataframe[narrative_column].fillna("").astype(str).str.strip()
    labels = dataframe[target_column].fillna("").astype(str).str.strip()
    valid_rows = narratives.ne("") & labels.ne("")
    filtered_narratives = narratives.loc[valid_rows].reset_index(drop=True)
    filtered_labels = labels.loc[valid_rows].reset_index(drop=True)
    excluded_rows = len(dataframe) - len(filtered_narratives)

    if filtered_narratives.empty:
        raise ValueError("No rows contain both a non-empty narrative and target label.")

    return filtered_narratives, filtered_labels, excluded_rows


def validate_stratified_split(
    labels: pd.Series,
    test_size: float,
) -> None:
    """Fail early with clear guidance when stratification cannot preserve classes."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    class_counts = labels.value_counts()
    rare_classes = class_counts[class_counts < 2]
    if not rare_classes.empty:
        examples = ", ".join(
            f"{label!r} ({count})" for label, count in rare_classes.items()
        )
        raise ValueError(
            "Stratified train/test splitting requires at least two examples per "
            f"class. Classes with too few examples: {examples}. Add data or define "
            "a documented rare-class grouping before training."
        )

    class_count = len(class_counts)
    test_rows = math.ceil(len(labels) * test_size)
    train_rows = len(labels) - test_rows
    if test_rows < class_count or train_rows < class_count:
        raise ValueError(
            f"The requested split leaves {train_rows} training and {test_rows} test "
            f"rows for {class_count} classes. Increase the dataset size or adjust "
            "test_size so each split can contain every class."
        )


def make_pipelines(random_seed: int = DEFAULT_RANDOM_SEED) -> dict[str, Pipeline]:
    """Build text-only pipelines with vectorization fitted inside each model."""
    # Keeping TF-IDF inside Pipeline ensures it only learns vocabulary and IDF
    # weights from the training portion after the stratified split.
    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=100_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def evaluate_baselines(
    dataframe: pd.DataFrame,
    narrative_column: str = DEFAULT_NARRATIVE_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Split labeled narratives, fit both models, and return comparable reports."""
    narratives, labels, excluded_rows = prepare_labeled_data(
        dataframe,
        narrative_column=narrative_column,
        target_column=target_column,
    )
    validate_stratified_split(labels, test_size)

    # Split row positions before fitting any vectorizer, selector, or classifier.
    train_positions, test_positions = train_test_split(
        np.arange(len(narratives)),
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    x_train = narratives.iloc[train_positions]
    x_test = narratives.iloc[test_positions]
    y_train = labels.iloc[train_positions]
    y_test = labels.iloc[test_positions]
    if "sample_row_id" in dataframe.columns:
        valid_rows = (
            dataframe[narrative_column].fillna("").astype(str).str.strip().ne("")
            & dataframe[target_column].fillna("").astype(str).str.strip().ne("")
        )
        row_ids = dataframe.loc[valid_rows, "sample_row_id"].reset_index(drop=True)
    else:
        row_ids = pd.Series(range(len(narratives)))
    pipelines = make_pipelines(random_seed)
    all_classes = sorted(labels.unique())
    results: dict[str, Any] = {
        "target_column": target_column,
        "n_rows_used": len(narratives),
        "n_rows_excluded": excluded_rows,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "random_seed": random_seed,
        "test_size": test_size,
        "classes": all_classes,
        "train_positions": train_positions.tolist(),
        "test_positions": test_positions.tolist(),
        "models": {},
    }

    for model_name, pipeline in pipelines.items():
        # Fit only on training narratives; the pipeline vectorizer learns here.
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        classifier = pipeline.named_steps["classifier"]

        # Probability scores support ROC-AUC for Logistic Regression; LinearSVC
        # decision scores are accepted by ROC-AUC without probability calibration.
        if model_name == "logistic_regression":
            class_scores = pipeline.predict_proba(x_test)
        else:
            class_scores = pipeline.decision_function(x_test)

        model_report = classification_report(
            y_test,
            predictions,
            labels=all_classes,
            output_dict=True,
            zero_division=0,
        )
        per_class_report = {
            class_name: model_report[class_name]
            for class_name in all_classes
        }
        if len(all_classes) == 2:
            binary_scores = (
                class_scores[:, 1]
                if getattr(class_scores, "ndim", 1) == 2
                else class_scores
            )
            macro_roc_auc = roc_auc_score(y_test, binary_scores)
        else:
            macro_roc_auc = roc_auc_score(
                y_test,
                class_scores,
                labels=classifier.classes_,
                multi_class="ovr",
                average="macro",
            )

        macro_metrics = {
            "macro_f1": f1_score(
                y_test, predictions, labels=all_classes, average="macro", zero_division=0
            ),
            "macro_precision": precision_score(
                y_test, predictions, labels=all_classes, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                y_test, predictions, labels=all_classes, average="macro", zero_division=0
            ),
            "macro_roc_auc_ovr": macro_roc_auc,
        }
        results["models"][model_name] = {
            "metrics": macro_metrics,
            "per_class_report": per_class_report,
            "classification_report": model_report,
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=all_classes,
            ).tolist(),
            "confusion_matrix_labels": all_classes,
            "weakest_classes": sorted(
                (
                    {
                        "class": class_name,
                        "f1_score": per_class_report[class_name]["f1-score"],
                        "support": int(per_class_report[class_name]["support"]),
                    }
                    for class_name in all_classes
                ),
                key=lambda class_result: (
                    class_result["f1_score"],
                    -class_result["support"],
                    class_result["class"],
                ),
            )[:5],
            "test_predictions": [
                {
                    "sample_row_id": int(row_ids.iloc[position]),
                    "narrative": narratives.iloc[position],
                    "actual": labels.iloc[position],
                    "predicted": str(prediction),
                }
                for position, prediction in zip(test_positions, predictions)
            ],
        }

    return results


def write_results(results: dict[str, Any], output_dir: Path) -> None:
    """Save readable metric/report JSON and one labeled confusion CSV per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "traditional_baseline_report.json"
    # Narratives are written only to a local, ignored CSV for paired error review.
    public_results = {
        **{
            key: value
            for key, value in results.items()
            if key not in {"train_positions", "test_positions"}
        },
        "models": {
            model_name: {
                key: value
                for key, value in model_result.items()
                if key != "test_predictions"
            }
            for model_name, model_result in results["models"].items()
        },
    }
    report_path.write_text(
        json.dumps(public_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for model_name, model_result in results["models"].items():
        pd.DataFrame(model_result["test_predictions"]).to_csv(
            output_dir / f"{model_name}_test_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        matrix = pd.DataFrame(
            model_result["confusion_matrix"],
            index=model_result["confusion_matrix_labels"],
            columns=model_result["confusion_matrix_labels"],
        )
        matrix.index.name = "actual"
        matrix.to_csv(output_dir / f"{model_name}_confusion_matrix.csv")


def print_results(results: dict[str, Any]) -> None:
    """Print macro metrics, per-class scores, and the least-recalled classes."""
    print(
        f"Rows used: {results['n_rows_used']:,}; "
        f"rows excluded: {results['n_rows_excluded']:,}; "
        f"train: {results['n_train']:,}; test: {results['n_test']:,}"
    )

    for model_name, model_result in results["models"].items():
        print(f"\n{model_name.replace('_', ' ').title()}")
        print("Macro metrics:")
        for metric_name, metric_value in model_result["metrics"].items():
            print(f"  {metric_name}: {metric_value:.4f}")

        print("\nPer-class classification report:")
        for class_name, class_metrics in model_result["per_class_report"].items():
            print(
                f"  {class_name}: precision={class_metrics['precision']:.4f}, "
                f"recall={class_metrics['recall']:.4f}, "
                f"f1={class_metrics['f1-score']:.4f}, "
                f"support={int(class_metrics['support'])}"
            )

        print("\nConfusion matrix (rows=actual, columns=predicted):")
        print(
            pd.DataFrame(
                model_result["confusion_matrix"],
                index=model_result["confusion_matrix_labels"],
                columns=model_result["confusion_matrix_labels"],
            ).to_string()
        )
        weakest_classes = ", ".join(
            f"{entry['class']} (F1={entry['f1_score']:.3f}, "
            f"n={entry['support']})"
            for entry in model_result["weakest_classes"]
        )
        print(f"\nWeakest classes by F1: {weakest_classes}")


def main() -> None:
    """Load the CFPB CSV and run the traditional text-classification baselines."""
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF Logistic Regression and Linear SVM models."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/raw/complaints.csv",
        help="CFPB CSV path (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Label column (default: Issue).",
    )
    parser.add_argument(
        "--narrative-column",
        default=DEFAULT_NARRATIVE_COLUMN,
        help="Narrative column (default: Consumer complaint narrative).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction reserved for test data (default: 0.2).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Fixed random seed for the split (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/baseline",
        help="Directory for JSON report and confusion matrices.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CFPB CSV file not found: {csv_path.resolve()}")

    # Read only the selected target and narrative, avoiding unnecessary PII columns.
    dataframe = pd.read_csv(
        csv_path,
        usecols=[args.narrative_column, args.target_column],
        encoding="utf-8-sig",
        dtype=str,
    )
    results = evaluate_baselines(
        dataframe,
        narrative_column=args.narrative_column,
        target_column=args.target_column,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    print_results(results)
    write_results(results, Path(args.output_dir))
    print(f"\nReports written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
