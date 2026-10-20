"""Tests for deterministic and approximately balanced sampling."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.balanced_sample import choose_balanced_classes, load_balanced_sample


class BalancedSampleTests(unittest.TestCase):
    """Check sample sizing, class balance, and repeatability."""

    def test_class_selection_chooses_largest_balanced_sample(self) -> None:
        class_names, rows_per_class = choose_balanced_classes(
            pd.Series({"A": 10, "B": 7, "C": 4, "Rare": 1}),
            target_size=12,
        )

        self.assertEqual(class_names, ["A", "B", "C"])
        self.assertEqual(rows_per_class, 4)

    def test_chunked_sample_is_balanced_and_reproducible(self) -> None:
        rows = [
            {
                "Consumer complaint narrative": f"Narrative {label} {row_number}",
                "Issue": label,
            }
            for label, count in (("A", 12), ("B", 10), ("C", 8))
            for row_number in range(count)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "complaints.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)

            first_sample = load_balanced_sample(
                csv_path,
                target_size=15,
                random_seed=42,
                chunksize=7,
            )
            second_sample = load_balanced_sample(
                csv_path,
                target_size=15,
                random_seed=42,
                chunksize=7,
            )

        self.assertEqual(len(first_sample), 15)
        self.assertEqual(first_sample["Issue"].value_counts().to_dict(), {
            "A": 5,
            "B": 5,
            "C": 5,
        })
        pd.testing.assert_frame_equal(first_sample, second_sample)

    def test_singleton_classes_are_not_included_in_a_stratifiable_sample(self) -> None:
        class_names, rows_per_class = choose_balanced_classes(
            pd.Series({"A": 6, "B": 6, "singleton": 1}),
            target_size=10,
        )

        self.assertEqual(class_names, ["A", "B"])
        self.assertEqual(rows_per_class, 5)


if __name__ == "__main__":
    unittest.main()
