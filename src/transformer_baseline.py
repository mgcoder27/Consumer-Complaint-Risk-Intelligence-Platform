"""Fine-tune DistilBERT and compare it on the traditional baseline's exact split."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Support direct script execution as well as package/module imports.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.balanced_sample import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    load_balanced_sample,
)
from src.traditional_baseline import (
    DEFAULT_TARGET_COLUMN,
    evaluate_baselines,
    write_results,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
DEFAULT_TEST_SIZE = 0.2


class ComplaintDataset(Dataset):
    """Hold tokenized narratives and integer labels for Hugging Face Trainer."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        """Return the number of complaint examples."""
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one padded model input and its training label."""
        item = {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def make_compute_metrics(class_count: int) -> Any:
    """Create the macro-F1 callback used for evaluation and best-checkpoint choice."""
    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        logits = evaluation_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted_labels = np.argmax(logits, axis=1)
        true_labels = evaluation_prediction.label_ids
        return {
            "f1_macro": float(
                f1_score(
                    true_labels,
                    predicted_labels,
                    labels=list(range(class_count)),
                    average="macro",
                    zero_division=0,
                )
            )
        }

    return compute_metrics


def create_training_arguments(
    output_dir: Path,
    random_seed: int,
    use_gpu: bool,
) -> TrainingArguments:
    """Set Colab-friendly training, evaluation, checkpoint, and precision options."""
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1,
        fp16=use_gpu,
        report_to="none",
        seed=random_seed,
        data_seed=random_seed,
        dataloader_num_workers=2 if use_gpu else 0,
    )


def prepare_transformer_data(
    sample: pd.DataFrame,
    train_positions: list[int],
    test_positions: list[int],
    tokenizer: Any,
    target_column: str = DEFAULT_TARGET_COLUMN,
    narrative_column: str = "Consumer complaint narrative",
    max_length: int = MAX_LENGTH,
) -> tuple[ComplaintDataset, ComplaintDataset, list[str]]:
    """Tokenize the exact same train/test positions used by the baseline."""
    labels = sorted(sample[target_column].astype(str).unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    encoded_labels = sample[target_column].astype(str).map(label_to_id).tolist()

    # Tokenize separately after the fixed split; no labels or text are shuffled here.
    train_texts = sample.iloc[train_positions][narrative_column].astype(str).tolist()
    test_texts = sample.iloc[test_positions][narrative_column].astype(str).tolist()
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        max_length=max_length,
    )
    train_labels = [encoded_labels[position] for position in train_positions]
    test_labels = [encoded_labels[position] for position in test_positions]
    return (
        ComplaintDataset(train_encodings, train_labels),
        ComplaintDataset(test_encodings, test_labels),
        labels,
    )


def compare_model_errors(
    baseline_predictions: list[dict[str, Any]],
    transformer_predictions: list[dict[str, Any]],
    output_dir: Path,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Write paired disagreement examples and return a short count-based analysis."""
    baseline_by_id = {
        int(row["sample_row_id"]): row for row in baseline_predictions
    }
    transformer_by_id = {
        int(row["sample_row_id"]): row for row in transformer_predictions
    }
    shared_ids = sorted(set(baseline_by_id) & set(transformer_by_id))
    if not shared_ids:
        raise ValueError("Baseline and transformer predictions have no shared test IDs.")

    examples: list[dict[str, Any]] = []
    transformer_right = 0
    baseline_right = 0
    both_right = 0
    both_wrong = 0
    for sample_row_id in shared_ids:
        baseline_row = baseline_by_id[sample_row_id]
        transformer_row = transformer_by_id[sample_row_id]
        actual_label = str(baseline_row["actual"])
        if actual_label != str(transformer_row["actual"]):
            raise ValueError(
                f"Model labels do not agree for sample row {sample_row_id}."
            )

        baseline_correct = str(baseline_row["predicted"]) == actual_label
        transformer_correct = str(transformer_row["predicted"]) == actual_label
        if transformer_correct and not baseline_correct:
            transformer_right += 1
            examples.append(
                {
                    "case_type": "Transformer correct; baseline wrong",
                    "sample_row_id": sample_row_id,
                    "actual_issue": actual_label,
                    "baseline_prediction": baseline_row["predicted"],
                    "transformer_prediction": transformer_row["predicted"],
                    "narrative": baseline_row["narrative"],
                }
            )
        elif baseline_correct and not transformer_correct:
            baseline_right += 1
            examples.append(
                {
                    "case_type": "Baseline correct; transformer wrong",
                    "sample_row_id": sample_row_id,
                    "actual_issue": actual_label,
                    "baseline_prediction": baseline_row["predicted"],
                    "transformer_prediction": transformer_row["predicted"],
                    "narrative": baseline_row["narrative"],
                }
            )
        elif baseline_correct:
            both_right += 1
        else:
            both_wrong += 1

    selected_examples = (
        [row for row in examples if row["case_type"] == "Transformer correct; baseline wrong"][
            :max_examples
        ]
        + [
            row for row in examples
            if row["case_type"] == "Baseline correct; transformer wrong"
        ][:max_examples]
    )
    pd.DataFrame(selected_examples).to_csv(
        output_dir / "paired_error_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "comparison_baseline": "best of Logistic Regression and Linear SVM by macro-F1",
        "test_examples": len(shared_ids),
        "transformer_only_correct": transformer_right,
        "baseline_only_correct": baseline_right,
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "example_rows_written": len(selected_examples),
        "interpretation": (
            "Review paired_error_examples.csv for narrative-level disagreements. "
            "It contains up to five cases each where the transformer alone is "
            "correct and where the baseline alone is correct. The counts describe "
            "this held-out split and do not establish why either model was correct."
        ),
    }


def train_and_compare(
    sample: pd.DataFrame,
    output_dir: Path,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    target_column: str = DEFAULT_TARGET_COLUMN,
    narrative_column: str = "Consumer complaint narrative",
    max_length: int = MAX_LENGTH,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    """Run the baseline and transformer on one balanced sample and identical split."""
    if "sample_row_id" not in sample.columns:
        raise ValueError("Balanced sample must include the sample_row_id column.")
    if sample["sample_row_id"].duplicated().any():
        raise ValueError("Balanced sample_row_id values must be unique.")

    # Evaluate the traditional models on precisely this sample and seed first.
    baseline_results = evaluate_baselines(
        sample,
        narrative_column=narrative_column,
        target_column=target_column,
        test_size=test_size,
        random_seed=random_seed,
    )
    best_baseline_name = max(
        baseline_results["models"],
        key=lambda name: baseline_results["models"][name]["metrics"]["macro_f1"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results(baseline_results, output_dir / "baseline")

    # The baseline exposes split positions so transformer examples cannot drift.
    train_positions = baseline_results["train_positions"]
    test_positions = baseline_results["test_positions"]
    labels = baseline_results["classes"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print(
            "Warning: CUDA GPU is not available. Training will use CPU and may be "
            "very slow; select a free T4 GPU runtime in Google Colab for the intended run."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, test_dataset, tokenized_labels = prepare_transformer_data(
        sample,
        train_positions,
        test_positions,
        tokenizer,
        target_column=target_column,
        narrative_column=narrative_column,
        max_length=max_length,
    )
    if tokenized_labels != labels:
        raise ValueError("Transformer label ordering does not match the baseline.")

    # Seed classifier-head initialization as well as Trainer sampling and dropout.
    set_seed(random_seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    training_arguments = create_training_arguments(
        output_dir,
        random_seed,
        use_gpu,
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(len(labels)),
    )
    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    transformer_class_ids = np.argmax(prediction_output.predictions, axis=1)
    transformer_predictions = [
        {
            "sample_row_id": int(sample.iloc[position]["sample_row_id"]),
            "actual": str(sample.iloc[position][target_column]),
            "predicted": id_to_label[int(predicted_class)],
        }
        for position, predicted_class in zip(test_positions, transformer_class_ids)
    ]
    actual_labels = [
        str(sample.iloc[position][target_column]) for position in test_positions
    ]
    predicted_labels = [row["predicted"] for row in transformer_predictions]
    transformer_f1 = float(
        f1_score(
            actual_labels,
            predicted_labels,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    transformer_report = classification_report(
        actual_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    transformer_matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=labels,
    )
    pd.DataFrame(
        transformer_predictions,
    ).to_csv(
        output_dir / "transformer_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        transformer_matrix,
        index=labels,
        columns=labels,
    ).rename_axis("actual").to_csv(
        output_dir / "transformer_confusion_matrix.csv",
    )
    (output_dir / "transformer_classification_report.json").write_text(
        json.dumps(transformer_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(output_dir / "best_model")

    # Compare three systems on identical actual labels and exact held-out rows.
    comparison_rows = [
        {
            "model": model_name_key.replace("_", " ").title(),
            "macro_f1": model_result["metrics"]["macro_f1"],
            "test_size": baseline_results["n_test"],
            "split_seed": random_seed,
            "sample_size": len(sample),
        }
        for model_name_key, model_result in baseline_results["models"].items()
    ]
    comparison_rows.append(
        {
            "model": "DistilBERT",
            "macro_f1": transformer_f1,
            "test_size": baseline_results["n_test"],
            "split_seed": random_seed,
            "sample_size": len(sample),
        }
    )
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(
        output_dir / "macro_f1_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison = compare_model_errors(
        baseline_results["models"][best_baseline_name]["test_predictions"],
        transformer_predictions,
        output_dir,
    )

    results = {
        "model_name": model_name,
        "target_column": target_column,
        "sample_size": len(sample),
        "sample_class_count": len(labels),
        "sample_manifest": sample.attrs.get("manifest"),
        "max_length": max_length,
        "random_seed": random_seed,
        "test_size": test_size,
        "train_size": baseline_results["n_train"],
        "test_rows": baseline_results["n_test"],
        "train_positions": train_positions,
        "test_positions": test_positions,
        "label_order": labels,
        "transformer_macro_f1": transformer_f1,
        "transformer_classification_report": transformer_report,
        "comparison_table": comparison_rows,
        "paired_error_analysis": comparison,
        "best_baseline_for_error_review": best_baseline_name,
        "baseline_macro_f1": {
            name: model["metrics"]["macro_f1"]
            for name, model in baseline_results["models"].items()
        },
    }
    (output_dir / "transformer_comparison_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results


def main() -> None:
    """Prepare/load the balanced sample and run training on Colab or local hardware."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT and compare it on the baseline split."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="data/raw/complaints.csv",
        help="Raw CFPB CSV (default: data/raw/complaints.csv).",
    )
    parser.add_argument(
        "--sample-csv",
        default="data/processed/transformer/balanced_sample.csv",
        help="Cached balanced sample CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/transformer/results",
        help="Directory for metrics, predictions, and the fine-tuned model.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    sample_path = Path(args.sample_csv)
    if sample_path.is_file():
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
        manifest_path = sample_path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            sample.attrs["manifest"] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
    else:
        sample = load_balanced_sample(
            Path(args.input_csv),
            target_size=args.sample_size,
            random_seed=args.random_seed,
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
        sample_path.with_suffix(".manifest.json").write_text(
            json.dumps(sample.attrs["manifest"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    results = train_and_compare(
        sample,
        Path(args.output_dir),
        test_size=args.test_size,
        random_seed=args.random_seed,
        max_length=args.max_length,
        model_name=args.model_name,
    )
    print("\nMacro-F1 comparison:")
    print(pd.DataFrame(results["comparison_table"]).to_string(index=False))
    print("\nPaired error summary:")
    print(json.dumps(results["paired_error_analysis"], indent=2))
    print(f"\nResults saved under {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
