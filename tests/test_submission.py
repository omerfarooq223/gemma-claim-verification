"""Unit tests for submission CSV creation and validation."""

import os
import tempfile
import unittest
from gemma_claim_verification.submission import create_submission_file, validate_submission_file


class TestSubmission(unittest.TestCase):
    def test_create_and_validate_valid_submission(self):
        predictions = [
            {"id": "t1", "prediction": "SUPPORTS"},
            {"id": "t2", "prediction": "REFUTES"},
            {"id": "t3", "prediction": "NOT_ENOUGH_INFO"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "submission.csv")
            create_submission_file(
                predictions,
                output_csv_path=out_csv,
                expected_ids=["t1", "t2", "t3"],
            )
            report = validate_submission_file(
                out_csv,
                expected_ids=["t1", "t2", "t3"],
                expected_count=3,
            )
            self.assertEqual(report["status"], "VALID")
            self.assertEqual(report["row_count"], 3)
            self.assertEqual(report["unique_ids"], 3)
            self.assertEqual(report["label_distribution"]["SUPPORTS"], 1)
            self.assertEqual(report["label_distribution"]["REFUTES"], 1)
            self.assertEqual(report["label_distribution"]["NOT_ENOUGH_INFO"], 1)

    def test_validate_submission_invalid_label(self):
        predictions = [
            {"id": "t1", "prediction": "SUPPORTS"},
            {"id": "t2", "prediction": "INVALID_LABEL"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "submission.csv")
            with self.assertRaises(ValueError):
                create_submission_file(predictions, output_csv_path=out_csv)

    def test_validate_submission_duplicate_id(self):
        predictions = [
            {"id": "t1", "prediction": "SUPPORTS"},
            {"id": "t1", "prediction": "REFUTES"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "submission.csv")
            with self.assertRaises(ValueError):
                create_submission_file(predictions, output_csv_path=out_csv)

    def test_validate_submission_order_mismatch(self):
        predictions = [
            {"id": "t2", "prediction": "SUPPORTS"},
            {"id": "t1", "prediction": "REFUTES"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "submission.csv")
            with self.assertRaises(ValueError):
                create_submission_file(
                    predictions,
                    output_csv_path=out_csv,
                    expected_ids=["t1", "t2"],
                )


if __name__ == "__main__":
    unittest.main()
