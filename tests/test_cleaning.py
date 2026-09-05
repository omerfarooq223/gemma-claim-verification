"""Unit tests for dataset cleaning and normalization pipeline."""

import unittest
from gemma_claim_verification.cleaning import (
    clean_dataset,
    clean_record,
    normalize_evidence_list,
    normalize_label,
    normalize_text,
)


class TestCleaning(unittest.TestCase):
    def test_normalize_text(self):
        raw = "  This   is \u00a0a   test\nwith  newlines.\t"
        cleaned = normalize_text(raw)
        self.assertEqual(cleaned, "This is a test with newlines.")
        self.assertEqual(normalize_text(None), "")

    def test_normalize_label_aliases(self):
        self.assertEqual(normalize_label("SUPPORTS"), "SUPPORTS")
        self.assertEqual(normalize_label("supports"), "SUPPORTS")
        self.assertEqual(normalize_label("SUPPORTED"), "SUPPORTS")
        self.assertEqual(normalize_label("REFUTES"), "REFUTES")
        self.assertEqual(normalize_label("refutes"), "REFUTES")
        self.assertEqual(normalize_label("Refutes"), "REFUTES")
        self.assertEqual(normalize_label("NOT_ENOUGH_INFO"), "NOT_ENOUGH_INFO")
        self.assertEqual(normalize_label("not enough info"), "NOT_ENOUGH_INFO")
        self.assertEqual(normalize_label("NEI"), "NOT_ENOUGH_INFO")
        self.assertIsNone(normalize_label("invalid_label"))
        self.assertIsNone(normalize_label(""))
        self.assertIsNone(normalize_label(None))

    def test_normalize_evidence_list(self):
        raw_evidence = [
            "  First passage.  ",
            "",
            "   ",
            "Second passage.",
            "First passage.",  # duplicate
        ]
        cleaned = normalize_evidence_list(raw_evidence)
        self.assertEqual(cleaned, ["First passage.", "Second passage."])

    def test_clean_record_valid(self):
        record = {
            "id": "rec_001",
            "claim": "  The sky is blue.  ",
            "evidence": ["Passage A", "Passage A", ""],
            "label": "supports",
        }
        cleaned = clean_record(record, is_training=True)
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned["id"], "rec_001")
        self.assertEqual(cleaned["claim"], "The sky is blue.")
        self.assertEqual(cleaned["evidence"], ["Passage A"])
        self.assertEqual(cleaned["label"], "SUPPORTS")

    def test_clean_record_invalid_label_dropped(self):
        record = {
            "id": "rec_002",
            "claim": "Some claim.",
            "evidence": ["Passage A"],
            "label": "",
        }
        self.assertIsNone(clean_record(record, is_training=True))

    def test_clean_dataset_pipeline(self):
        raw_data = [
            {"id": "1", "claim": "Claim A", "evidence": ["Ev 1"], "label": "SUPPORTS"},
            {"id": "2", "claim": "Claim A", "evidence": ["Ev 1"], "label": "supports"},
            {"id": "3", "claim": "Claim B", "evidence": ["Ev 2"], "label": "SUPPORTS"},
            {"id": "4", "claim": "Claim B", "evidence": ["Ev 2"], "label": "REFUTES"},
            {"id": "5", "claim": "Claim C", "evidence": ["Ev 3"], "label": "unusable_label"},
        ]

        cleaned, report = clean_dataset(raw_data, is_training=True)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["id"], "1")
        self.assertEqual(report["raw_rows"], 5)
        self.assertEqual(report["dropped_missing_or_invalid_label"], 1)
        self.assertEqual(report["dropped_exact_duplicates"], 1)
        self.assertEqual(report["dropped_conflicting_duplicates"], 2)
        self.assertEqual(report["cleaned_rows"], 1)


if __name__ == "__main__":
    unittest.main()
