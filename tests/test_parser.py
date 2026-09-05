"""Unit tests for model output parsing and label extraction."""

import unittest
from gemma_claim_verification.inference import parse_model_prediction


class TestParser(unittest.TestCase):
    def test_parse_exact_final(self):
        pred, valid = parse_model_prediction("FINAL: SUPPORTS")
        self.assertEqual(pred, "SUPPORTS")
        self.assertTrue(valid)

        pred, valid = parse_model_prediction("FINAL: REFUTES")
        self.assertEqual(pred, "REFUTES")
        self.assertTrue(valid)

        pred, valid = parse_model_prediction("FINAL: NOT_ENOUGH_INFO")
        self.assertEqual(pred, "NOT_ENOUGH_INFO")
        self.assertTrue(valid)

    def test_parse_with_surrounding_text(self):
        text = "Based on the evidence, the statement is contradicted.\nFINAL: REFUTES\nThank you."
        pred, valid = parse_model_prediction(text)
        self.assertEqual(pred, "REFUTES")
        self.assertTrue(valid)

    def test_parse_multiple_final_occurrences(self):
        # The parser must select the LAST valid occurrence
        text = "Initially thought FINAL: SUPPORTS, but upon closer inspection:\nFINAL: REFUTES"
        pred, valid = parse_model_prediction(text)
        self.assertEqual(pred, "REFUTES")
        self.assertTrue(valid)

    def test_parse_fallback_without_final_prefix(self):
        text = "The evidence supports the claim."
        pred, valid = parse_model_prediction(text)
        self.assertEqual(pred, "SUPPORTS")
        self.assertFalse(valid)

    def test_parse_unparseable_output(self):
        text = "I cannot determine anything from this sentence."
        pred, valid = parse_model_prediction(text)
        self.assertEqual(pred, "NOT_ENOUGH_INFO")
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
