import unittest

from gemma_claim_verification.training import compute_fixed_budget_plan


class FixedBudgetPlanTest(unittest.TestCase):

    def test_real935_budget(self):
        plan = compute_fixed_budget_plan(
            num_samples=935,
            max_optimizer_steps=136,
            batch_size=1,
            gradient_accumulation_steps=16,
        )

        self.assertEqual(plan["optimizer_steps"], 136)
        self.assertEqual(plan["micro_batches"], 2176)
        self.assertEqual(plan["example_exposures"], 2176)
        self.assertEqual(plan["full_passes"], 2)
        self.assertEqual(plan["partial_pass_examples"], 306)

    def test_augmented1085_budget(self):
        plan = compute_fixed_budget_plan(
            num_samples=1085,
            max_optimizer_steps=136,
            batch_size=1,
            gradient_accumulation_steps=16,
        )

        self.assertEqual(plan["optimizer_steps"], 136)
        self.assertEqual(plan["micro_batches"], 2176)
        self.assertEqual(plan["example_exposures"], 2176)
        self.assertEqual(plan["full_passes"], 2)
        self.assertEqual(plan["partial_pass_examples"], 6)

    def test_invalid_budget_rejected(self):
        with self.assertRaises(ValueError):
            compute_fixed_budget_plan(
                num_samples=935,
                max_optimizer_steps=0,
                batch_size=1,
                gradient_accumulation_steps=16,
            )


if __name__ == "__main__":
    unittest.main()
