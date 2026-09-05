#!/usr/bin/env python3
"""CLI script to audit and clean training data."""

import argparse
import json
import sys
from gemma_claim_verification.cleaning import clean_dataset
from gemma_claim_verification.data import load_jsonl, save_jsonl
from gemma_claim_verification.hashing import compute_sha256


def main():
    parser = argparse.ArgumentParser(description="Clean and audit raw claim-verification datasets.")
    parser.add_argument("--input", "-i", required=True, help="Path to input raw JSONL file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save cleaned JSONL file.")
    parser.add_argument("--is_training", action="store_true", default=True, help="Apply training-level label filtering and duplicate conflict checks.")
    args = parser.parse_args()

    print(f"Loading raw data from: {args.input}")
    input_sha = compute_sha256(args.input)
    print(f"Input SHA-256: {input_sha}")

    records = load_jsonl(args.input)
    print(f"Raw records loaded: {len(records)}")

    cleaned_records, audit_report = clean_dataset(records, is_training=args.is_training)

    save_jsonl(cleaned_records, args.output)
    output_sha = compute_sha256(args.output)

    print("\n--- Audit Report ---")
    print(json.dumps(audit_report, indent=2))
    print(f"\nCleaned dataset saved to: {args.output}")
    print(f"Cleaned dataset SHA-256:  {output_sha}")


if __name__ == "__main__":
    main()
