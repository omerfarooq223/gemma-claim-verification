"""Deterministic inference engine and robust output parser."""

import re
from typing import Any, Dict, List, Optional, Tuple
from .constants import ALLOWED_LABELS, ALLOWED_LABELS_SET, DEFAULT_MAX_NEW_TOKENS, LabelEnum
from .prompts import format_chat_message

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


FINAL_LABEL_PATTERN = re.compile(
    r"(?i)\bFINAL\s*:\s*(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)\b"
)

FALLBACK_LABEL_PATTERN = re.compile(
    r"(?i)\b(SUPPORTS|REFUTES|NOT_ENOUGH_INFO)\b"
)


def parse_model_prediction(raw_text: str) -> Tuple[str, bool]:
    """Parse raw generation text to extract final classification label.

    Returns:
        Tuple of (predicted_label, is_valid)
    """
    if not raw_text or not raw_text.strip():
        return LabelEnum.NOT_ENOUGH_INFO.value, False

    clean_text = raw_text.strip()

    # Find all matches for "FINAL: <LABEL>"
    final_matches = FINAL_LABEL_PATTERN.findall(clean_text)
    if final_matches:
        # Return the LAST valid occurrence as specified
        last_match = final_matches[-1].upper()
        if last_match in ALLOWED_LABELS_SET:
            return last_match, True

    # Conservative fallback if no explicit FINAL: prefix was found
    fallback_matches = FALLBACK_LABEL_PATTERN.findall(clean_text)
    if fallback_matches:
        last_fallback = fallback_matches[-1].upper()
        if last_fallback in ALLOWED_LABELS_SET:
            return last_fallback, False

    return LabelEnum.NOT_ENOUGH_INFO.value, False


def predict_single(
    model: Any,
    processor: Any,
    claim: str,
    evidence: Any,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run deterministic inference on a single claim-evidence pair."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch must be installed for model inference.")

    if device is None:
        device = next(model.parameters()).device

    messages = format_chat_message(claim, evidence)
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )

    # Slice only the newly generated tokens
    prompt_len = inputs.input_ids.shape[1]
    new_tokens = generated_ids[0, prompt_len:]
    raw_output = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    pred_label, is_valid = parse_model_prediction(raw_output)

    return {
        "raw_output": raw_output,
        "prediction": pred_label,
        "is_valid": is_valid,
    }


def predict_dataset(
    model: Any,
    processor: Any,
    records: List[Dict[str, Any]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run batch-wise or iterative deterministic prediction on a list of records."""
    results = []
    device = next(model.parameters()).device

    for idx, item in enumerate(records):
        res = predict_single(
            model=model,
            processor=processor,
            claim=item["claim"],
            evidence=item.get("evidence", []),
            max_new_tokens=max_new_tokens,
            device=device,
        )
        record_res = {
            "id": item["id"],
            "claim": item["claim"],
            "prediction": res["prediction"],
            "raw_output": res["raw_output"],
            "is_valid": res["is_valid"],
        }
        if "label" in item:
            record_res["true_label"] = item["label"]
        results.append(record_res)

        if progress_callback:
            progress_callback(idx + 1, len(records))

    return results
