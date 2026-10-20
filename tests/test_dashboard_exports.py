"""Tests for the Power BI dashboard export aggregates."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.dashboard_exports import export_dashboard_csvs


class DashboardExportTests(unittest.TestCase):
    """Check the exported rates, rankings, missing-month logic, and caveats."""

    def test_exports_aggregates_from_gzip_and_joins_population(self) -> None:
        complaints = pd.DataFrame(
            {
                "Date received": [
                    "2024-01-05",
                    "2024-01-15",
                    "2024-03-01",
                    "2024-03-12",
                    "2024-03-29",
                ],
                "Product": [
                    "Credit card",
                    "Credit card",
                    "Debt collection",
                    "Mortgage",
                    "Credit reporting",
                ],
                "Issue": [
                    "Billing disputes",
                    "Unauthorized transaction",
                    "Communication tactics",
                    "Application processing delay",
                    "Incorrect information on your report",
                ],
                "Sub-issue": ["", "", "", "", ""],
                "Company": ["Alpha\u00a0", "Alpha", "Beta", "Beta", "Beta"],
                "State": ["CA", "California", "NY", "NY", "New York"],
                "Company response to consumer": [
                    "Closed with monetary relief",
                    "Closed with non-monetary relief",
                    "Closed with monetary relief",
                    "In progress",
                    "Closed with explanation",
                ],
            }
        )
        population = pd.DataFrame(
            {
                "NAME": ["California", "New York"],
                "POPESTIMATE2024": [100_000, 200_000],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            input_path = folder / "complaints.csv.gz"
            population_path = folder / "population.csv"
            output_dir = folder / "powerbi"
            complaints.to_csv(input_path, index=False, compression="gzip")
            population.to_csv(population_path, index=False)

            summary = export_dashboard_csvs(
                input_path,
                output_dir,
                population_csv=population_path,
                population_state_column="NAME",
                population_column="POPESTIMATE2024",
                chunksize=2,
            )

            monthly = pd.read_csv(output_dir / "monthly_volume_trend.csv")
            categories = pd.read_csv(output_dir / "risk_category_summary.csv")
            companies = pd.read_csv(output_dir / "company_relief_rates.csv")
            states = pd.read_csv(output_dir / "state_complaints_per_capita.csv")
            recommendations = pd.read_csv(
                output_dir / "business_recommendations.csv"
            )
            model_drivers = pd.read_csv(output_dir / "model_drivers.csv")

        self.assertEqual(summary["source_rows"], 5)
        self.assertEqual(monthly["month"].tolist(), ["2024-01", "2024-02", "2024-03"])
        self.assertEqual(monthly["complaint_count"].tolist(), [2, 0, 3])
        self.assertEqual(int(monthly.loc[0, "monetary_relief_count"]), 1)
        self.assertEqual(int(monthly.loc[1, "month_over_month_change"]), -2)

        alpha = companies.loc[companies["company"] == "Alpha"].iloc[0]
        self.assertEqual(int(alpha["complaint_count"]), 2)
        self.assertEqual(int(alpha["closed_outcome_count"]), 2)
        self.assertAlmostEqual(alpha["monetary_relief_rate_pct"], 50.0)
        self.assertIn("Fraud/Identity Theft", set(categories["risk_category"]))
        self.assertNotIn("Unknown", set(categories["risk_category"]))
        self.assertEqual(int(categories["complaint_count"].sum()), len(complaints))

        california = states.loc[states["state"] == "CA"].iloc[0]
        self.assertEqual(int(california["population"]), 100_000)
        self.assertAlmostEqual(california["complaints_per_100k"], 2.0)
        self.assertEqual(len(recommendations), 4)
        self.assertTrue(recommendations["evidence"].str.contains(r"\d").all())
        self.assertIn("No trained model-driver artifact", model_drivers.loc[0, "status"])

    def test_per_capita_values_remain_blank_without_population_data(self) -> None:
        complaints = pd.DataFrame(
            {
                "Date received": ["2024-01-05"],
                "Product": ["Credit card"],
                "Issue": ["Billing disputes"],
                "Company": ["Alpha"],
                "State": ["CA"],
                "Company response to consumer": ["Closed with explanation"],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            folder = Path(temporary_directory)
            input_path = folder / "complaints.csv"
            output_dir = folder / "powerbi"
            complaints.to_csv(input_path, index=False)

            export_dashboard_csvs(input_path, output_dir, chunksize=1)
            states = pd.read_csv(output_dir / "state_complaints_per_capita.csv")

        self.assertTrue(pd.isna(states.loc[0, "population"]))
        self.assertTrue(pd.isna(states.loc[0, "complaints_per_100k"]))


if __name__ == "__main__":
    unittest.main()
