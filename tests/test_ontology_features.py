"""Tests for the custom risk ontology and state population feature."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ontology_features import (
    RISK_CATEGORIES,
    add_ontology_and_population_features,
    load_state_populations,
    map_risk_category,
    normalize_state,
    process_complaints_csv,
)


class OntologyFeatureTests(unittest.TestCase):
    """Use synthetic records to check categories and per-capita calculations."""

    def test_maps_high_risk_categories_with_specific_precedence(self) -> None:
        self.assertEqual(
            map_risk_category(
                "Credit card",
                "Charged for a purchase or transfer I did not make",
                "Unauthorized transaction",
            )[0],
            "Fraud/Identity Theft",
        )
        self.assertEqual(
            map_risk_category("Debt collection", "Communication tactics")[0],
            "Debt Collection Misconduct",
        )
        self.assertEqual(
            map_risk_category(
                "Credit reporting",
                "Incorrect information on your report",
            )[0],
            "Credit Reporting Error",
        )
        self.assertEqual(
            map_risk_category("Credit card", "Billing disputes")[0],
            "Billing Dispute",
        )
        self.assertEqual(
            map_risk_category("Mortgage", "Application processing delay")[0],
            "Other",
        )

    def test_normalizes_state_names_postal_codes_and_fips(self) -> None:
        self.assertEqual(normalize_state("California"), "CA")
        self.assertEqual(normalize_state(" ca "), "CA")
        self.assertEqual(normalize_state("06"), "CA")
        self.assertEqual(normalize_state("District of Columbia"), "DC")
        self.assertIsNone(normalize_state("United States"))

    def test_population_join_adds_state_count_and_rate(self) -> None:
        complaints = pd.DataFrame(
            {
                "Product": ["Credit card", "Debt collection", "Mortgage"],
                "Issue": ["Billing disputes", "Communication tactics", "Other"],
                "Sub-issue": [None, None, None],
                "State": ["CA", "California", "XX"],
            }
        )

        featured = add_ontology_and_population_features(
            complaints,
            state_complaint_counts={"CA": 2},
            state_populations={"CA": 40_000_000},
        )

        self.assertEqual(
            featured["risk_category"].tolist(),
            ["Billing Dispute", "Debt Collection Misconduct", "Other"],
        )
        self.assertEqual(featured.loc[0, "state_complaint_count"], 2)
        self.assertEqual(featured.loc[0, "state_population"], 40_000_000)
        self.assertAlmostEqual(float(featured.loc[0, "complaints_per_100k"]), 0.005)
        self.assertTrue(pd.isna(featured.loc[2, "complaints_per_100k"]))
        self.assertEqual(
            set(featured["risk_category"]),
            {
                "Billing Dispute",
                "Debt Collection Misconduct",
                "Other",
            },
        )

    def test_csv_process_counts_all_rows_and_writes_rate_features(self) -> None:
        complaint_rows = pd.DataFrame(
            {
                "Product": ["Credit reporting", "Credit card", "Mortgage"],
                "Issue": ["Incorrect information on report", "Billing disputes", "Other"],
                "State": ["California", "CA", "NY"],
            }
        )
        population_rows = pd.DataFrame(
            {"NAME": ["California", "New York"], "POPESTIMATE2024": [40_000_000, 20_000_000]}
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            complaints_path = folder / "complaints.csv"
            population_path = folder / "population.csv"
            output_path = folder / "enriched.csv"
            complaint_rows.to_csv(complaints_path, index=False)
            population_rows.to_csv(population_path, index=False)

            row_count = process_complaints_csv(
                complaints_path,
                population_path,
                output_path,
                state_population_column="NAME",
                population_column="POPESTIMATE2024",
                chunksize=2,
            )
            enriched = pd.read_csv(output_path)

        self.assertEqual(row_count, 3)
        self.assertEqual(enriched["state_complaint_count"].tolist()[:2], [2, 2])
        self.assertTrue((enriched["risk_category"] != "").all())
        self.assertAlmostEqual(enriched.loc[0, "complaints_per_100k"], 0.005)

    def test_duplicate_normalized_population_states_raise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            population_path = Path(temporary_directory) / "population.csv"
            pd.DataFrame(
                {"state": ["CA", "California"], "population": [100, 100]}
            ).to_csv(population_path, index=False)

            with self.assertRaisesRegex(ValueError, "one row per state"):
                load_state_populations(population_path, "state", "population")


if __name__ == "__main__":
    unittest.main()
