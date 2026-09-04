"""Training loop and exact QLoRA fine-tuning execution."""

import math
import os
import random
from typing import Any, Dict, List, Optional
from .constants import DEFAULT_MAX_SEQ_LENGTH
from .prompts import format_chat_message, format_completion_target

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_cosine_schedule_with_warmup
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class ClaimVerificationDataset(Dataset):
    """PyTorch Dataset for completion-only supervised fine-tuning."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        processor: Any,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
    ):
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        claim = item["claim"]
        evidence = item["evidence"]
        label = item["label"]

        # User messages format
        messages = format_chat_message(claim, evidence)
        target_text = format_completion_target(label)

        # Apply chat template for prompt
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        full_text = prompt_text + target_text

        # Tokenize prompt and full sequence
        prompt_tokens = self.processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        full_tokens = self.processor.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=False,  # Assert no truncation permitted
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = full_tokens.input_ids[0]
        attention_mask = full_tokens.attention_mask[0]

        if len(input_ids) > self.max_length:
            raise ValueError(
                f"Sequence length ({len(input_ids)}) exceeds max_length ({self.max_length}) for item {item.get('id')}"
            )

        # Mask prompt tokens in labels with -100 (completion-only loss)
        labels = input_ids.clone()
        prompt_len = prompt_tokens.input_ids.shape[1]
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate single-example or batch items with left/right padding."""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": padded_inputs,
        "attention_mask": padded_mask,
        "labels": padded_labels,
    }


def train_qlora(
    model: Any,
    processor: Any,
    train_records: List[Dict[str, Any]],
    output_dir: str = "checkpoints/final_adapter",
    epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    seed: int = 42,
    max_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> str:
    """Execute exact competition training recipe for Gemma 4 QLoRA."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and Transformers must be installed.")

    os.makedirs(output_dir, exist_ok=True)

    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model training preflight
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Prepare Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_samples = len(train_records)
    steps_per_epoch = math.ceil(num_samples / (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Starting QLoRA training: {num_samples} records, {epochs} epochs, {total_steps} total optimizer steps.")

    global_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, epochs + 1):
        # Deterministic per-epoch shuffle
        epoch_rng = random.Random(seed + epoch)
        shuffled_indices = list(range(num_samples))
        epoch_rng.shuffle(shuffled_indices)
        epoch_records = [train_records[i] for i in shuffled_indices]

        dataset = ClaimVerificationDataset(epoch_records, processor, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} completed — Avg Loss: {avg_loss:.4f} (Global Step: {global_step})")

    # Save final adapter and tokenizer configurations
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training complete. Adapter saved to {output_dir}")
    return output_dir
