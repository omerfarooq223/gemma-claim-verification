"""Dataset loading, validation, batching, and PyTorch dataset definitions."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from .constants import ALLOWED_LABELS_SET
from .cleaning import clean_dataset
from .prompts import format_chat_message, format_completion_target


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset JSONL not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} of {file_path}: {e}")
    return records


def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """Save a list of dictionaries to a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare_training_curriculum(
    real_clean_path: str,
    contrastive_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Combine clean real data with audited contrastive data."""
    real_records = load_jsonl(real_clean_path)
    if contrastive_path and os.path.isfile(contrastive_path):
        contrastive_records = load_jsonl(contrastive_path)
        combined = real_records + contrastive_records
    else:
        combined = real_records

    # Validate all entries
    for idx, item in enumerate(combined):
        if "claim" not in item or "evidence" not in item or "label" not in item:
            raise ValueError(f"Item {idx} missing required fields (claim, evidence, label)")
        if item["label"] not in ALLOWED_LABELS_SET:
            raise ValueError(f"Item {idx} has invalid label: '{item['label']}'")

    return combined
