"""Gemma 4 Claim Verification Package.

Reliable evidence-based three-way claim verification using fine-tuned Google Gemma 4 models with QLoRA.
"""

__version__ = "1.0.0"

from .constants import (
    ALLOWED_LABELS,
    LabelEnum,
    PROMPT_TEMPLATE,
    SYSTEM_INSTRUCTION,
    TARGET_MODULES,
    KNOWN_HASHES,
)

__all__ = [
    "ALLOWED_LABELS",
    "LabelEnum",
    "PROMPT_TEMPLATE",
    "SYSTEM_INSTRUCTION",
    "TARGET_MODULES",
    "KNOWN_HASHES",
]
