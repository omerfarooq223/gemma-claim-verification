"""Prompt formatting and Gemma chat template builders."""

from typing import Any, Dict, List, Optional, Union
import unicodedata
from .constants import PROMPT_TEMPLATE, SYSTEM_INSTRUCTION, ALLOWED_LABELS_SET


def format_evidence_passages(evidence: Union[str, List[str]]) -> str:
    """Format single string or list of evidence passages into numbered format."""
    if isinstance(evidence, str):
        evidence_list = [evidence.strip()] if evidence.strip() else []
    elif isinstance(evidence, list):
        evidence_list = [
            str(e).strip() for e in evidence if str(e).strip()
        ]
    else:
        evidence_list = []

    if not evidence_list:
        return "[1] No evidence provided."

    formatted_passages = []
    for idx, passage in enumerate(evidence_list, start=1):
        # Normalize whitespace inside passage
        clean_passage = " ".join(passage.split())
        formatted_passages.append(f"[{idx}] {clean_passage}")

    return "\n".join(formatted_passages)


def build_claim_prompt(claim: str, evidence: Union[str, List[str]]) -> str:
    """Build user prompt string with system instructions, claim, and evidence."""
    # Normalize unicode
    norm_claim = unicodedata.normalize("NFKC", str(claim).strip())
    norm_claim = " ".join(norm_claim.split())

    formatted_evidence = format_evidence_passages(evidence)

    return PROMPT_TEMPLATE.format(
        instruction=SYSTEM_INSTRUCTION,
        claim=norm_claim,
        evidence=formatted_evidence,
    )


def format_chat_message(claim: str, evidence: Union[str, List[str]]) -> List[Dict[str, str]]:
    """Format single user turn for Gemma chat template."""
    content = build_claim_prompt(claim, evidence)
    return [{"role": "user", "content": content}]


def format_completion_target(label: str) -> str:
    """Format the supervised assistant target string."""
    norm_label = str(label).strip()
    if norm_label not in ALLOWED_LABELS_SET:
        raise ValueError(f"Invalid label for completion target: '{label}'. Must be one of {ALLOWED_LABELS_SET}")
    return f"FINAL: {norm_label}"
