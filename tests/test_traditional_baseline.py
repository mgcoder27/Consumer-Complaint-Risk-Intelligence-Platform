"""Focused tests for the traditional issue-classification baseline."""

import unittest

import pandas as pd
from sklearn.pipeline import Pipeline

from src.traditional_baseline import (
    evaluate_baselines,
    make_pipelines,
    prepare_labeled_data,
    validate_stratified_split,
)


class TraditionalBaselineTests(unittest.TestCase):
    """Check filtering, stratification guards, pipeline configuration, and metrics."""

    def make_small_dataset(self) -> pd.DataFrame:
        """Create balanced synthetic data with distinct text per issue."""
        examples = []
        issue_terms = {
            "Fraud": "unauthorized transfer stolen funds",
            "Billing": "incorrect bill fee charge",
            "Credit reporting": "credit report score bureau",
        }
        for issue, phrase in issue_terms.items():
            for number in range(10):
                examples.append(
                    {
                        "Consumer complaint narrative": f"{phrase} example {number}",
                        "Issue": issue,
                    }
                )
        return pd.DataFrame(examples)

    def test_prepare_drops_blank_narratives_and_blank_labels(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Consumer complaint narrative": ["text", " ", "other"],
                "Issue": ["Fraud", "Fraud", ""],
            }
        )

        narratives, labels, excluded_rows = prepare_labeled_data(dataframe)

        self.assertEqual(narratives.tolist(), ["text"])
        self.assertEqual(labels.tolist(), ["Fraud"])
        self.assertEqual(excluded_rows, 2)

    def test_stratification_rejects_singleton_class_with_guidance(self) -> None:
        labels = pd.Series(["Fraud", "Fraud", "Billing"])

        with self.assertRaisesRegex(ValueError, "at least two examples per class"):
            validate_stratified_split(labels, test_size=0.2)

    def test_pipelines_include_class_balancing_and_tfidf(self) -> None:
        pipelines = make_pipelines()

        self.assertEqual(set(pipelines), {"logistic_regression", "linear_svm"})
        for pipeline in pipelines.values():
            self.assertIsInstance(pipeline, Pipeline)
            self.assertEqual(pipeline.named_steps["classifier"].class_weight, "balanced")
            self.assertEqual(pipeline.named_steps["tfidf"].ngram_range, (1, 2))

    def test_evaluation_returns_macro_and_per_class_metrics(self) -> None:
        results = evaluate_baselines(
            self.make_small_dataset(),
            test_size=0.3,
            random_seed=42,
        )

        self.assertEqual(results["n_rows_used"], 30)
        self.assertEqual(results["n_train"], 21)
        self.assertEqual(results["n_test"], 9)
        self.assertEqual(set(results["models"]), {"logistic_regression", "linear_svm"})
        for model_result in results["models"].values():
            self.assertIn("macro_f1", model_result["metrics"])
            self.assertIn("macro_precision", model_result["metrics"])
            self.assertIn("macro_recall", model_result["metrics"])
            self.assertIn("macro_roc_auc_ovr", model_result["metrics"])
            self.assertEqual(len(model_result["confusion_matrix"]), 3)
            self.assertEqual(len(model_result["weakest_classes"]), 3)


if __name__ == "__main__":
    unittest.main()
