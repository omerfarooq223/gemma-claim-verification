#!/usr/bin/env python3
"""CLI script to train Gemma 4 12B with QLoRA."""

import argparse
import os
import sys
import yaml

from gemma_claim_verification.constants import DEFAULT_MAX_SEQ_LENGTH
from gemma_claim_verification.data import load_jsonl, prepare_training_curriculum
from gemma_claim_verification.hashing import compute_sha256
from gemma_claim_verification.modeling import build_lora_model, load_base_model_and_processor
from gemma_claim_verification.training import train_qlora


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 using QLoRA.")
    parser.add_argument("--config", "-c", default="configs/final_train.yaml", help="Path to training YAML config.")
    parser.add_argument("--train_clean", required=True, help="Path to audited clean real dataset JSONL (e.g. 935 rows).")
    parser.add_argument("--contrastive", default=None, help="Optional path to audited contrastive dataset JSONL (e.g. 150 rows).")
    parser.add_argument("--output_dir", "-o", default="checkpoints/final_adapter", help="Directory to save trained LoRA adapter.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config from: {args.config}")
    print(f"Real clean data:    {args.train_clean} (SHA-256: {compute_sha256(args.train_clean)})")
    if args.contrastive:
        print(f"Contrastive data:   {args.contrastive} (SHA-256: {compute_sha256(args.contrastive)})")

    train_records = prepare_training_curriculum(args.train_clean, args.contrastive)
    print(f"Total training curriculum size: {len(train_records)} examples")

    model_name = cfg["model"]["name"]
    load_4bit = cfg["model"].get("load_in_4bit", True)
    max_length = cfg["model"].get("max_length", DEFAULT_MAX_SEQ_LENGTH)

    print(f"\nLoading base model: {model_name} (4-bit NF4: {load_4bit})")
    base_model, processor = load_base_model_and_processor(
        model_name_or_path=model_name,
        load_in_4bit=load_4bit,
    )

    lora_cfg = cfg["lora"]
    print(f"Attaching LoRA adapter: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}, dropout={lora_cfg['dropout']}")
    model = build_lora_model(
        base_model=base_model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg.get("target_modules"),
    )

    t_cfg = cfg["training"]
    train_qlora(
        model=model,
        processor=processor,
        train_records=train_records,
        output_dir=args.output_dir,
        epochs=t_cfg.get("epochs", 2),
        batch_size=t_cfg.get("batch_size", 1),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(t_cfg.get("warmup_ratio", 0.05)),
        seed=t_cfg.get("seed", 42),
        max_length=max_length,
    )


if __name__ == "__main__":
    main()
