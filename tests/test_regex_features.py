"""Focused tests for regex narrative cleaning and feature extraction."""

import unittest

from src.regex_features import clean_narrative, extract_narrative_features


class NarrativeFeatureTests(unittest.TestCase):
    """Check feature values on small, safe synthetic narratives."""

    def test_extracts_currency_date_account_card_and_masked_token(self) -> None:
        features = extract_narrative_features(
            "On 03/14/2024 I paid $1,250.00. My card ending XXXX-1234 was used."
        )

        self.assertEqual(features["amount_mentioned"], 1250.0)
        self.assertEqual(features["amount_mentions"], "1,250.00")
        self.assertTrue(features["has_amount"])
        self.assertEqual(features["date_mentions"], "03/14/2024")
        self.assertTrue(features["has_date"])
        self.assertTrue(features["has_card_mention"])
        self.assertEqual(features["n_masked_tokens"], 1)

    def test_extracts_multiple_amounts_and_month_name_date(self) -> None:
        features = extract_narrative_features(
            "An account was charged USD 2,000 and $35 on January 5, 2025."
        )

        self.assertEqual(features["amount_mentioned"], 2000.0)
        self.assertEqual(features["amount_mentions"], "2,000 | 35")
        self.assertEqual(features["n_amount_mentions"], 2)
        self.assertEqual(features["date_mentions"], "January 5, 2025")
        self.assertTrue(features["has_account_mention"])

    def test_cleanup_decodes_entities_removes_markup_and_normalizes_spaces(self) -> None:
        cleaned = clean_narrative("<p>Bank&nbsp; &amp; card</p>\n\tissue")

        self.assertEqual(cleaned, "Bank & card issue")

    def test_no_matches_use_clear_empty_values(self) -> None:
        features = extract_narrative_features("A short narrative without markers.")

        self.assertIsNone(features["amount_mentioned"])
        self.assertFalse(features["has_amount"])
        self.assertEqual(features["date_mentions"], "")
        self.assertFalse(features["has_date"])
        self.assertEqual(features["n_masked_tokens"], 0)
        self.assertEqual(features["narrative_length"], len(features["narrative_clean"]))

    def test_empty_input_is_supported(self) -> None:
        features = extract_narrative_features(None)

        self.assertEqual(features["narrative_clean"], "")
        self.assertEqual(features["narrative_length"], 0)


if __name__ == "__main__":
    unittest.main()
