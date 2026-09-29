import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from gemma_claim_verification.training import train_qlora


class DummyTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=None,
        truncation=False,
    ):
        # Keep tokenization deliberately tiny.
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        )


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        return "prompt"

    def save_pretrained(self, output_dir):
        Path(output_dir, "processor.saved").write_text(
            "ok",
            encoding="utf-8",
        )


if _TORCH_AVAILABLE:
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(use_cache=True)
            self.forward_calls = 0

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
        ):
            self.forward_calls += 1

            # Scalar differentiable loss; enough to exercise backward/optimizer.
            loss = self.weight * 0.0 + 1.0

            return SimpleNamespace(loss=loss)

        def save_pretrained(self, output_dir):
            Path(output_dir, "model.saved").write_text(
                "ok",
                encoding="utf-8",
            )
else:
    DummyModel = None  # type: ignore


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def make_records(n):
    labels = [
        "SUPPORTS",
        "REFUTES",
        "NOT_ENOUGH_INFO",
    ]

    return [
        {
            "id": f"row_{i}",
            "claim": f"claim {i}",
            "evidence": [f"evidence {i}"],
            "label": labels[i % 3],
        }
        for i in range(n)
    ]


@unittest.skipIf(not _TORCH_AVAILABLE, "PyTorch not installed")
class FixedBudgetTrainingTest(unittest.TestCase):
    def run_case(self, n_rows):
        model = DummyModel()
        processor = DummyProcessor()

        optimizer_step_calls = 0
        scheduler = CountingScheduler()

        real_adamw = torch.optim.AdamW

        def optimizer_factory(*args, **kwargs):
            nonlocal optimizer_step_calls

            optimizer = real_adamw(*args, **kwargs)
            original_step = optimizer.step

            def counted_step(*step_args, **step_kwargs):
                nonlocal optimizer_step_calls
                optimizer_step_calls += 1
                return original_step(*step_args, **step_kwargs)

            optimizer.step = counted_step
            return optimizer

        def scheduler_factory(*args, **kwargs):
            return scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "gemma_claim_verification.training.torch.optim.AdamW",
                side_effect=optimizer_factory,
            ), patch(
                "gemma_claim_verification.training.get_cosine_schedule_with_warmup",
                side_effect=scheduler_factory,
            ):
                train_qlora(
                    model=model,
                    processor=processor,
                    train_records=make_records(n_rows),
                    output_dir=tmpdir,
                    epochs=99,  # Must be ignored in fixed-step mode.
                    max_optimizer_steps=3,
                    batch_size=1,
                    gradient_accumulation_steps=4,
                    learning_rate=2e-4,
                    weight_decay=0.01,
                    warmup_ratio=0.0,
                    seed=42,
                    max_length=256,
                )

            self.assertTrue(
                Path(tmpdir, "model.saved").exists()
            )
            self.assertTrue(
                Path(tmpdir, "processor.saved").exists()
            )

        # 3 optimizer steps * 4 micro-batches per step.
        self.assertEqual(model.forward_calls, 12)
        self.assertEqual(optimizer_step_calls, 3)
        self.assertEqual(scheduler.step_calls, 3)

    def test_fixed_budget_with_five_row_dataset(self):
        # 12 exposures = 2 complete passes + 2 examples.
        self.run_case(5)

    def test_fixed_budget_with_seven_row_dataset(self):
        # 12 exposures = 1 complete pass + 5 examples.
        self.run_case(7)


if __name__ == "__main__":
    unittest.main()
