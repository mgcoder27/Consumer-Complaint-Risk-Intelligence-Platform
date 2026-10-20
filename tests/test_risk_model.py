"""Tests for relief-label rules and leak-free risk-model feature construction."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.risk_model import (
    FORBIDDEN_FEATURE_COLUMNS,
    derive_relief_target,
    sample_modeling_rows,
    prepare_risk_features,
)


class ReliefRiskFeatureTests(unittest.TestCase):
    """Check labels, unresolved outcomes, and pre-outcome predictor construction."""

    def test_target_requires_closed_outcome_and_marks_monetary_relief(self) -> None:
        self.assertEqual(
            derive_relief_target("Closed with monetary relief")[0],
            1,
        )
        self.assertEqual(
            derive_relief_target("Closed with non-monetary relief")[0],
            0,
        )
        self.assertEqual(
            derive_relief_target("Closed with non monetary relief")[0],
            0,
        )
        self.assertEqual(
            derive_relief_target("Closed with monetary relief")[0],
            1,
        )
        self.assertEqual(derive_relief_target("Closed with explanation")[0], 0)
        self.assertIsNone(derive_relief_target("In progress")[0])
        self.assertIsNone(derive_relief_target(None)[0])
        self.assertIsNone(derive_relief_target("Unknown response")[0])
        self.assertIsNone(derive_relief_target("Closed with relief")[0])
        self.assertIsNone(derive_relief_target("Closed")[0])
        self.assertIsNone(derive_relief_target("Untimely response")[0])

    def test_vectorized_sampling_labels_match_scalar_rules(self) -> None:
        from src.risk_model import _derive_relief_targets

        responses = pd.Series(
            [
                "Closed with monetary relief",
                "Closed with non-monetary relief",
                "Closed with explanation",
                "Closed without relief",
                "Closed",
                "Closed with relief",
                "In progress",
                "Untimely response",
                None,
            ]
        )
        vector_targets, _ = _derive_relief_targets(responses)
        scalar_targets = [
            derive_relief_target(response)[0] for response in responses
        ]

        self.assertEqual(
            [None if pd.isna(target) else int(target) for target in vector_targets],
            scalar_targets,
        )

    def test_large_file_sampler_preserves_target_prevalence_and_detects_no_text(self) -> None:
        rows = [
            {
                "Company response to consumer": (
                    "Closed with monetary relief"
                    if index < 10
                    else "Closed with explanation"
                ),
                "Product": "Credit card",
                "Issue": "Billing disputes",
                "State": "CA",
            }
            for index in range(60)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "complaints.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            sample, manifest = sample_modeling_rows(
                csv_path,
                max_rows=30,
                chunksize=13,
                random_seed=42,
            )

        self.assertEqual(len(sample), 30)
        self.assertEqual(manifest["sample_class_counts"], {
            "no_monetary_relief": 25,
            "monetary_relief": 5,
        })
        self.assertFalse(manifest["narrative_column_available"])

    def test_features_exclude_response_and_post_outcome_fields(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Consumer complaint narrative": [
                    "I did not authorize the $125 card charge on 01/02/2024 XXXX.",
                    "I disputed a fee on my account.",
                    "The complaint is still under review.",
                ],
                "Company response to consumer": [
                    "Closed with monetary relief",
                    "Closed with explanation",
                    "In progress",
                ],
                "Product": ["Credit card", "Credit card", "Credit card"],
                "Issue": [
                    "Unauthorized transaction",
                    "Billing disputes",
                    "Billing disputes",
                ],
                "Sub-issue": ["", "", ""],
                "State": ["ca", "NY", "CA"],
                "Timely response?": ["Yes", "No", "No"],
                "Consumer disputed?": ["No", "Yes", "No"],
            }
        )

        features, labels, metadata = prepare_risk_features(dataframe)

        self.assertEqual(labels.tolist(), [1, 0])
        self.assertEqual(metadata["excluded_rows"], 1)
        self.assertEqual(metadata["class_counts"], {
            "no_monetary_relief": 1,
            "monetary_relief": 1,
        })
        self.assertTrue(
            set(FORBIDDEN_FEATURE_COLUMNS).isdisjoint(features.columns)
        )
        self.assertIn("narrative_clean", features.columns)
        self.assertIn("n_masked_tokens", features.columns)
        self.assertIn("risk_category", features.columns)
        self.assertIn("Product", features.columns)
        self.assertIn("State", features.columns)
        self.assertEqual(features["risk_category"].tolist(), [
            "Fraud/Identity Theft",
            "Billing Dispute",
        ])
        self.assertEqual(features["State"].tolist(), ["CA", "NY"])

    def test_missing_narrative_column_is_reported_without_fake_text_features(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Company response to consumer": [
                    "Closed with monetary relief",
                    "Closed with explanation",
                ],
                "Product": ["Credit card", "Mortgage"],
                "Issue": ["Billing disputes", "Application processing delay"],
                "State": ["CA", "NY"],
            }
        )

        features, labels, metadata = prepare_risk_features(dataframe)

        self.assertEqual(labels.tolist(), [1, 0])
        self.assertFalse(metadata["narrative_column_available"])
        self.assertFalse(metadata["regex_features_available"])
        self.assertEqual(features["narrative_clean"].tolist(), ["", ""])
        self.assertEqual(features["narrative_length"].tolist(), [0, 0])
        self.assertTrue(
            set(FORBIDDEN_FEATURE_COLUMNS).isdisjoint(features.columns)
        )


if __name__ == "__main__":
    unittest.main()
