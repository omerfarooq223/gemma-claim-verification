#!/usr/bin/env python3
"""CLI script to run deterministic inference on a test set."""

import argparse
import os
import sys
import yaml

from gemma_claim_verification.constants import DEFAULT_MAX_NEW_TOKENS
from gemma_claim_verification.data import load_jsonl
from gemma_claim_verification.hashing import compute_sha256
from gemma_claim_verification.inference import predict_dataset
from gemma_claim_verification.modeling import load_adapter, load_base_model_and_processor
from gemma_claim_verification.submission import create_submission_file


def main():
    parser = argparse.ArgumentParser(description="Run deterministic claim verification inference.")
    parser.add_argument("--test", "-t", required=True, help="Path to test JSONL file.")
    parser.add_argument("--adapter", "-a", required=True, help="Path to fine-tuned LoRA adapter directory.")
    parser.add_argument("--config", "-c", default="configs/final_inference.yaml", help="Path to inference YAML config.")
    parser.add_argument("--output", "-o", default="outputs/submission.csv", help="Path to save submission CSV.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_sha = compute_sha256(args.test)
    print(f"Loading test records from: {args.test} (SHA-256: {test_sha})")
    records = load_jsonl(args.test)
    print(f"Loaded {len(records)} test examples.")

    model_name = cfg["model"]["name"]
    load_4bit = cfg["model"].get("load_in_4bit", True)
    max_new_tokens = cfg.get("generation", {}).get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)

    print(f"Loading base model: {model_name}")
    base_model, processor = load_base_model_and_processor(
        model_name_or_path=model_name,
        load_in_4bit=load_4bit,
    )

    print(f"Attaching LoRA adapter from: {args.adapter}")
    adapter_weights_file = os.path.join(args.adapter, "adapter_model.safetensors")
    if os.path.isfile(adapter_weights_file):
        adapter_sha = compute_sha256(adapter_weights_file)
        print(f"Adapter weight SHA-256: {adapter_sha}")

    model = load_adapter(base_model, args.adapter, is_trainable=False)

    print(f"\nRunning deterministic inference ({len(records)} items)...")
    predictions = predict_dataset(
        model=model,
        processor=processor,
        records=records,
        max_new_tokens=max_new_tokens,
    )

    invalid_count = sum(1 for p in predictions if not p["is_valid"])
    print(f"Inference complete. Invalid outputs: {invalid_count}")

    expected_ids = [r["id"] for r in records]
    create_submission_file(
        predictions=predictions,
        output_csv_path=args.output,
        expected_ids=expected_ids,
    )


if __name__ == "__main__":
    main()
