#!/usr/bin/env bash

# ==============================================================================
# Script: make_15_commits.sh
# Author: Muhammad Umar Farooq
# Description: Stages and creates 15 meaningful, atomic git commits for the
#              gemma-claim-verification repository.
# ==============================================================================

set -e

echo "🚀 Starting 15 Staged Commits for Gemma Claim Verification..."

# Commit 1: Core Repo Setup
echo "Creating Commit 1/15..."
git add .gitignore LICENSE pyproject.toml requirements.txt src/gemma_claim_verification/__init__.py
git commit -m "chore: initialize repository structure, license, and dependencies"

# Commit 2: Constants & Enums
echo "Creating Commit 2/15..."
git add src/gemma_claim_verification/constants.py
git commit -m "feat(core): implement core domain constants, label mappings, and verification schemas"

# Commit 3: Prompts
echo "Creating Commit 3/15..."
git add src/gemma_claim_verification/prompts.py
git commit -m "feat(prompts): define canonical evidence-grounded prompt templates"

# Commit 4: Cleaning
echo "Creating Commit 4/15..."
git add src/gemma_claim_verification/cleaning.py
git commit -m "feat(cleaning): build 10-step unicode normalization and semantic audit pipeline"

# Commit 5: Data & Schema
echo "Creating Commit 5/15..."
git add src/gemma_claim_verification/data.py data/
git commit -m "feat(data): add JSONL dataset loader and contrastive curriculum utilities"

# Commit 6: Modeling
echo "Creating Commit 6/15..."
git add src/gemma_claim_verification/modeling.py
git commit -m "feat(modeling): implement Gemma 4 12B NF4 model setup and PEFT configuration"

# Commit 7: Training & Configs
echo "Creating Commit 7/15..."
git add src/gemma_claim_verification/training.py configs/final_train.yaml
git commit -m "feat(training): implement response-only loss QLoRA trainer and hyperparameter config"

# Commit 8: Inference
echo "Creating Commit 8/15..."
git add src/gemma_claim_verification/inference.py configs/final_inference.yaml
git commit -m "feat(inference): implement greedy deterministic inference engine"

# Commit 9: Evaluation
echo "Creating Commit 9/15..."
git add src/gemma_claim_verification/evaluation.py
git commit -m "feat(evaluation): add macro-F1 metrics, confusion matrix, and error analysis"

# Commit 10: Submission & Hashes
echo "Creating Commit 10/15..."
git add src/gemma_claim_verification/hashing.py src/gemma_claim_verification/submission.py outputs/
git commit -m "feat(submission): add SHA-256 integrity verification and submission builder"

# Commit 11: CLI Scripts
echo "Creating Commit 11/15..."
git add scripts/clean_training_data.py scripts/train_qlora.py scripts/predict.py scripts/evaluate.py scripts/verify_artifact.py
git commit -m "feat(cli): add CLI scripts for cleaning, training, evaluation, and verification"

# Commit 12: Tests
echo "Creating Commit 12/15..."
git add tests/
git commit -m "test: add unit test suite for cleaning, prompts, parser, and submission assertions"

# Commit 13: Docs & Notebooks
echo "Creating Commit 13/15..."
git add docs/ artifacts/ notebooks/
git commit -m "docs: add methodology reports, data audit docs, and experiment lineage"

# Commit 14: Hugging Face Deployment & App
echo "Creating Commit 14/15..."
git add app.py HUGGINGFACE_GUIDE.md
git commit -m "feat(hf): add Gradio web app for HF Spaces and HF deployment manual"

# Commit 15: Polished README & Daily Guide
echo "Creating Commit 15/15..."
git add README.md DAILY_PUSH_GUIDE.md scripts/make_15_commits.sh
git commit -m "docs(readme): polish README celebrating AI Seekho Day 2026 1st Place Win"

echo "✅ All 15 commits successfully created in local Git history!"
echo "Run 'git log --oneline -n 15' to inspect your clean commit history."
