"""Unit tests for prompt formatting and template logic."""

import unittest
from gemma_claim_verification.prompts import (
    build_claim_prompt,
    format_chat_message,
    format_completion_target,
    format_evidence_passages,
)


class TestPrompt(unittest.TestCase):
    def test_format_evidence_passages(self):
        single_str = "Evidence text single."
        formatted = format_evidence_passages(single_str)
        self.assertEqual(formatted, "[1] Evidence text single.")

        multi_list = ["First item", "Second item"]
        formatted_multi = format_evidence_passages(multi_list)
        self.assertEqual(formatted_multi, "[1] First item\n[2] Second item")

        empty_list = []
        formatted_empty = format_evidence_passages(empty_list)
        self.assertEqual(formatted_empty, "[1] No evidence provided.")

    def test_build_claim_prompt(self):
        claim = "Gemma 4 is released by Google."
        evidence = ["Google DeepMind developed the Gemma model family.", "Gemma 4 was released in 2026."]
        prompt = build_claim_prompt(claim, evidence)

        # Check that system instructions and label definitions exist
        self.assertIn("SUPPORTS: the evidence establishes the claim.", prompt)
        self.assertIn("REFUTES: the evidence contradicts the claim.", prompt)
        self.assertIn("NOT_ENOUGH_INFO: the evidence neither establishes nor contradicts the specific claim.", prompt)

        # Check exact final output instruction
        self.assertIn("End your response exactly as:\nFINAL: SUPPORTS\nor\nFINAL: REFUTES\nor\nFINAL: NOT_ENOUGH_INFO", prompt)

        # Check claim and numbered evidence
        self.assertIn("Claim:\nGemma 4 is released by Google.", prompt)
        self.assertIn("Evidence:\n[1] Google DeepMind developed the Gemma model family.\n[2] Gemma 4 was released in 2026.", prompt)

    def test_format_completion_target(self):
        self.assertEqual(format_completion_target("SUPPORTS"), "FINAL: SUPPORTS")
        self.assertEqual(format_completion_target("REFUTES"), "FINAL: REFUTES")
        self.assertEqual(format_completion_target("NOT_ENOUGH_INFO"), "FINAL: NOT_ENOUGH_INFO")

        with self.assertRaises(ValueError):
            format_completion_target("UNKNOWN_LABEL")


if __name__ == "__main__":
    unittest.main()
