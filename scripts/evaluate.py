#!/usr/bin/env python3
"""CLI script to evaluate predictions against gold labels."""

import argparse
import json
import os
import sys

from gemma_claim_verification.data import load_jsonl
from gemma_claim_verification.evaluation import compute_metrics, format_evaluation_summary
from gemma_claim_verification.hashing import compute_sha256


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted claims against gold ground truth.")
    parser.add_argument("--gold", "-g", required=True, help="Path to gold JSONL file containing true labels.")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions JSONL or CSV file.")
    parser.add_argument("--output_json", "-o", default=None, help="Optional path to save metrics JSON.")
    args = parser.parse_args()

    gold_records = load_jsonl(args.gold)
    gold_dict = {r["id"]: r["label"] for r in gold_records if "id" in r and "label" in r}

    print(f"Loaded {len(gold_dict)} gold labels from {args.gold} (SHA-256: {compute_sha256(args.gold)})")

    # Load predictions
    pred_dict = {}
    if args.predictions.endswith(".jsonl"):
        pred_records = load_jsonl(args.predictions)
        for r in pred_records:
            pred_dict[r["id"]] = r.get("prediction", r.get("label"))
    elif args.predictions.endswith(".csv"):
        import csv
        with open(args.predictions, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pred_dict[row["id"]] = row["label"]
    else:
        raise ValueError("Unsupported predictions format. Must be .jsonl or .csv")

    print(f"Loaded {len(pred_dict)} predictions from {args.predictions} (SHA-256: {compute_sha256(args.predictions)})")

    common_ids = [k for k in gold_dict if k in pred_dict]
    if len(common_ids) != len(gold_dict):
        print(f"Warning: Only {len(common_ids)} of {len(gold_dict)} gold IDs were present in predictions.")

    y_true = [gold_dict[i] for i in common_ids]
    y_pred = [pred_dict[i] for i in common_ids]

    metrics = compute_metrics(y_true, y_pred)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(format_evaluation_summary(metrics))

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics JSON to {args.output_json}")


if __name__ == "__main__":
    main()
