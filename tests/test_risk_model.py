"""Tests for relief-label rules and leak-free risk-model feature construction."""

import unittest

import pandas as pd

from src.risk_model import (
    FORBIDDEN_FEATURE_COLUMNS,
    derive_relief_target,
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


if __name__ == "__main__":
    unittest.main()
