# 🏆 1st Place Winner — AI Seekho Day 2026 Hackathon
# Reliable Evidence-Based Claim Verification with Gemma 4 12B

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)](tests/)
[![Base Model](https://img.shields.io/badge/Base_Model-google%2Fgemma--4--12B--it-orange.svg)](https://huggingface.co/google/gemma-4-12B-it)
[![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-QLoRA%20%2F%204--bit%20NF4-purple.svg)](#training-recipe--qlora-configuration)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Adapter-yellow.svg)](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo%20UI-blue.svg)](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier)
[![Author](https://img.shields.io/badge/Author-Muhammad%20Umar%20Farooq-blue.svg)](https://github.com/omerfarooq223)

</div>

---

> [!IMPORTANT]
> **🎉 CHAMPIONSHIP SOLUTION**: Built by **Muhammad Umar Farooq**, this system achieved **1st Place 🏆** in the official **AI Seekho Day 2026** competition. Starting from 1,000 noisy labeled examples, we engineered a 10-step semantic data audit pipeline, constructed a 935-example clean training dataset, synthesized 150 targeted contrastive trios, and fine-tuned **Gemma 4 12B** using **4-bit NF4 QLoRA**. 
>
> On the official **500-example supervised event-day benchmark**, our frozen checkpoint achieved **94.40% Accuracy** (472/500), **94.38% Macro-F1**, and **0 invalid outputs**.

<div align="center">
  <img src="docs/certificate.png" alt="AI Seekho Day 2026 1st Place Certificate - Muhammad Umar Farooq" width="85%" style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />
  <p><em>Official 1st Place Certificate of Participation & Victory — AI Seekho Day 2026</em></p>
</div>

---

## 📋 Table of Contents
- [Problem Overview & Objective](#-problem-overview--objective)
- [System Architecture](#-system-architecture)
- [Key Benchmarks & Evaluation](#-key-benchmarks--evaluation)
- [Data Engineering & Curriculum](#-data-engineering--curriculum)
- [Training Recipe & QLoRA Configuration](#-training-recipe--qlora-configuration)
- [Exact Prompt Formulation](#-exact-prompt-formulation)
- [Installation & Setup](#-installation--setup)
- [Quickstart & CLI Commands](#-quickstart--cli-commands)
- [Hugging Face Model & Live Demo App](#-hugging-face-model--live-demo-app)
- [Reproducibility & Cryptographic Hashes](#-reproducibility--cryptographic-hashes)
- [Key Engineering Insights](#-key-engineering-insights)
- [Repository Structure](#-repository-structure)
- [License & Citation](#-license--citation)

---

## 🎯 Problem Overview & Objective

Given a natural language **CLAIM** and one or more supplied **EVIDENCE** passages, the system classifies the factual relationship into exactly one of three canonical labels:

- **`SUPPORTS`**: The supplied evidence directly establishes the truth of the claim.
- **`REFUTES`**: The supplied evidence directly contradicts the claim.
- **`NOT_ENOUGH_INFO`**: The supplied evidence neither establishes nor contradicts the claim.

### Strict Evidence-Only Grounding
The model is supervised to reason **exclusively over the provided evidence passages**, ignoring prior parametric associations. This prevents hallucination and guarantees verifiable, evidence-grounded predictions.

---

## 🏗️ System Architecture

```
                       Raw Train (1,000 examples)
                                  │
                                  ▼
                   10-Step Audit & Semantic Cleaning
                                  │
                                  ▼
                       Clean Real (935 examples)
                                  │
                                  ├────────────────────────┐
                                  ▼                        ▼
                       Failure-Driven Analysis    Audited Contrastive Trios
                                  │                   (150 examples)
                                  │                        │
                                  └────────────┬───────────┘
                                               ▼
                                 Training Curriculum (1,085 examples)
                                               │
                                               ▼
                                    Fresh Gemma 4 12B-it Base
                                  ┌────────────────────────┐
                                  │ 4-bit NF4 Quantization │
                                  │ FP16 Compute Precision │
                                  │ Fresh LoRA Rank r=8    │
                                  │ Response-Only Loss     │
                                  └────────────────────────┘
                                               │
                                               ▼
                                   Frozen Selected Adapter
                        (SHA-256: 76630ec4620ff7244f3b6c9ef0350...)
                                               │
                                               ▼
                                   Deterministic Inference
                             (Greedy do_sample=False, beams=1)
                                               │
                                               ▼
                                  94.40% Accuracy / 94.38% F1
                                  (Supervised Test500 Evaluation)
```

---

## 📊 Key Benchmarks & Evaluation

### Official Benchmark Progression

| Evaluation Split | Examples | Accuracy | Macro-F1 | Invalid Outputs | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Organizer Dev Validation** | 300 | **92.33%** (277/300) | **92.27%** | **0** | Primary development signal (balanced 100/class) |
| **External Stress30** | 30 | **93.33%** (28/30) | **93.64%** | **0** | Out-of-distribution adversarial challenge |
| **Blind120 Holdout** | 120 | **93.33%** (112/120) | **93.37%** | **0** | Independent blind evaluation |
| **Event-Day Supervised Test500** | **500** | **94.40%** (472/500) | **94.38%** | **0** | **Official 1st Place Winning Benchmark 🏆** |

### Detailed Performance Breakdown (Test500)

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **`SUPPORTS`** | 86.39% | 98.80% | **92.18%** | 167 |
| **`REFUTES`** | 99.30% | 84.43% | **91.26%** | 167 |
| **`NOT_ENOUGH_INFO`** | 99.40% | 100.00% | **99.70%** | 166 |
| **Macro Average** | **95.03%** | **94.41%** | **94.38%** | **500** |

### Test500 Confusion Matrix

$$\text{Rows = Ground Truth}, \quad \text{Columns = Model Prediction}$$

```
                SUPPORTS    REFUTES    NOT_ENOUGH_INFO
SUPPORTS          165          1              1
REFUTES            26        141              0
NOT_ENOUGH_INFO     0          0            166
```

> [!NOTE]
> **Error Analysis**: The primary remaining error pattern was $26 \text{ REFUTES} \rightarrow \text{SUPPORTS}$ misclassifications. While the system achieved $99.40\%$ precision on `REFUTES` and $100.00\%$ recall on `NOT_ENOUGH_INFO`, subtle numerical or scope contradictions with supportive phrasing occasionally biased predictions toward entailment.

---

## 🧹 Data Engineering & Curriculum

Data quality was prioritized over blind synthetic volume expansion:

1. **10-Step Audit Pipeline**:
   - Unicode NFKC normalization and whitespace collapse.
   - Label alias normalization (`supports`, `refutes`, `NEI`, `SUPPORTED` $\rightarrow$ canonical forms).
   - Intra-example passage deduplication and empty passage filtering.
   - Elimination of 10 unusable/missing labels, 2 conflicting duplicate groups, and 53 exact duplicate rows.
   - **Result**: 1,000 raw rows $\rightarrow$ **935 audited clean examples**.

2. **Audited Contrastive Curriculum (150 Examples)**:
   - Built 225 contrastive trios (same evidence with 3 claim variants generating `SUPPORTS`, `REFUTES`, and `NOT_ENOUGH_INFO`) focusing on numerical inversions, entity swapping, and partial evidence.
   - **Phrase-Aware Auditor Verification**: Re-audited generated contrastive examples using a phrase-aware parser, achieving **60/60 agreement** on financial phrasing and validating the **225/225 semantic audit**.
   - Partitioned into **150 training contrastive examples** + **75 frozen holdout examples**.

$$\text{Final Training Curriculum} = 935\text{ Real Clean} + 150\text{ Audited Contrastive} = \mathbf{1,085\text{ Examples}}$$

---

## ⚙️ Training Recipe & QLoRA Configuration

| Hyperparameter | Value | Description |
|---|---|---|
| **Base Model** | `google/gemma-4-12B-it` | Google Gemma 4 12B instruction-tuned base |
| **Quantization** | 4-bit NF4 (`bnb_4bit_use_double_quant=True`) | Frozen base weights |
| **Compute Precision** | `torch.float16` | Forward/backward compute precision |
| **LoRA Rank / Alpha** | $r = 8, \quad \alpha = 16$ | Low-rank adapter matrix dimension |
| **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Language layers |
| **Trainable Parameters** | 32,784,384 | $\sim 0.27\%$ of base model parameters |
| **Optimizer** | AdamW (`lr=2e-4`, `weight_decay=0.01`) | Adaptive gradient optimization |
| **Batch Size** | Batch Size 1, Grad Accum 16 | **Effective Batch Size = 16** |
| **Epochs & Steps** | 2 Epochs (68 steps/epoch) | **136 total optimizer steps** |
| **Selected Adapter SHA-256** | `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e` | Winning checkpoint |

---

## 📝 Exact Prompt Formulation

```text
Classify the claim using only the supplied evidence.

SUPPORTS: the evidence establishes the claim.
REFUTES: the evidence contradicts the claim.
NOT_ENOUGH_INFO: the evidence neither establishes nor contradicts the specific claim.

End your response exactly as:
FINAL: SUPPORTS
or
FINAL: REFUTES
or
FINAL: NOT_ENOUGH_INFO

Claim:
<claim>

Evidence:
[1] <evidence passage 1>
[2] <evidence passage 2>
```

---

## 💻 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/omerfarooq223/gemma-claim-verification.git
cd gemma-claim-verification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

---

## 🚀 Quickstart & CLI Commands

### 1. Data Cleaning
```bash
python scripts/clean_training_data.py \
    --input data/competition/train.jsonl \
    --output data/derived/train_clean_935.jsonl
```

### 2. QLoRA Model Fine-Tuning
```bash
python scripts/train_qlora.py \
    --config configs/final_train.yaml \
    --train_clean data/derived/train_clean_935.jsonl \
    --contrastive data/derived/d4_contrastive_train_v1.jsonl \
    --output_dir checkpoints/final_adapter
```

### 3. Deterministic Inference
```bash
python scripts/predict.py \
    --test data/examples/sample.jsonl \
    --adapter checkpoints/final_adapter \
    --config configs/final_inference.yaml \
    --output outputs/submission.csv
```

### 4. Metric Evaluation & Verification
```bash
python scripts/evaluate.py \
    --gold data/examples/sample.jsonl \
    --predictions outputs/submission.csv \
    --output_json outputs/metrics.json
```

---

## 🤗 Hugging Face Model & Live Demo App

### Model Adapter Weights
The fine-tuned QLoRA adapter is published on Hugging Face:
- **Model Adapter**: [`omerfarooq223/gemma-4-12b-evidence-verification-qlora`](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora)
- **Base Model**: `google/gemma-4-12B-it`

```python
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load 4-bit NF4 Base Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained("google/gemma-4-12B-it")
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-12B-it",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load Winning LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    "omerfarooq223/gemma-4-12b-evidence-verification-qlora",
)
```

### Interactive Web App (Gradio)
Experience the model live on Hugging Face Spaces:
- 🌐 **Live Interactive Demo**: [`https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier`](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier)

To run locally:
```bash
python app.py
```
For complete deployment instructions to **Hugging Face Spaces**, see [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md).

---

## 🔒 Reproducibility & Cryptographic Hashes

| Artifact Key | File / Component | SHA-256 Checksum |
|---|---|---|
| `selected_final_adapter` | `adapter_model.safetensors` (Winning Checkpoint) | `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e` |
| `d4_champion_adapter` | `adapter_model.safetensors` (D4 Finalist) | `03b134b92e46f44b3802bf563668436250078a2ffef175e35c0c3c5b5ccf42d7` |
| `d1_champion_adapter` | `adapter_model.safetensors` (D1 Baseline) | `2c64a87e227e0638be9dc73e021f877f2536876ccf94cc116e04d60075164f8f` |
| `final_submission_csv` | `submission.csv` (Official Test500 Predictions) | `d4df310da9eb95205daf6bc9ad1f8cc874f7afe5f6d432ce8373f0a9f88dd012` |
| `clean_train935` | `train_clean_v3_semantic_recovered.jsonl` | `b706dbf0c0b4aab4cbcd07bb89c5d018f7c41e47a55b757016ef3d07a9713337` |
| `contrastive_train150` | `d4_contrastive_train_v1.jsonl` | `17258cbc0e40f4ebd1cd4d583e3a331e59217d62cf4ff36741e8b1e3a7a98f41` |

Verify checksums using the utility script:
```bash
python scripts/verify_artifact.py outputs/submission.csv --key final_submission_csv
```

---

## 💡 Key Engineering Insights

1. **Data Quality Over Raw Volume**: Systematic cleaning and semantic auditing produced immediate gains that fine-tuning on noisy data could never achieve.
2. **Targeted Contrastive Curriculum Beats Indiscriminate Expansion**: Adding 180 broad synthetic examples degraded validation accuracy by 2.8–3.4%, whereas 150 failure-driven contrastive trios boosted accuracy from 88.0% to 92.3%.
3. **Model Scale + Prompt Discipline**: Upgrading from 2B to 12B parameters provided superior reasoning, but strict prompt structure and completion-only loss masking were essential to eliminate invalid outputs.
4. **Validation Stability & Selection**: A subsequent reproduction run scored higher on dev validation (93.33%) but regressed on stress tests (86.67%). We retained the robust `76630...` adapter checkpoint.

---

## 📂 Repository Structure

```
gemma-claim-verification/
├── README.md                           # Master documentation & benchmark results
├── LICENSE                             # Apache 2.0 License
├── pyproject.toml                      # Package installation config
├── requirements.txt                    # Project dependencies
├── app.py                              # Gradio web application for HF Spaces
├── HUGGINGFACE_GUIDE.md                # Deployment manual for Hugging Face
├── DAILY_PUSH_GUIDE.md                 # 2-commit per day schedule guide
├── configs/
│   ├── final_train.yaml                # QLoRA fine-tuning hyperparameters
│   └── final_inference.yaml            # Deterministic greedy decoding parameters
├── src/
│   └── gemma_claim_verification/
│       ├── __init__.py
│       ├── constants.py                # Enums, prompt templates, known SHA hashes
│       ├── data.py                     # JSONL dataset loaders & PyTorch Dataset
│       ├── cleaning.py                 # 10-step semantic data auditing pipeline
│       ├── prompts.py                  # Prompt formatting & chat templates
│       ├── modeling.py                 # 4-bit NF4 loading & PEFT LoRA configuration
│       ├── training.py                 # Response-only loss trainer loop
│       ├── inference.py                # Greedy deterministic prediction engine
│       ├── evaluation.py               # Classification metrics & confusion matrix
│       ├── submission.py               # Submission CSV generator & assertions
│       └── hashing.py                  # Cryptographic SHA-256 verification
├── scripts/
│   ├── clean_training_data.py          # Data audit CLI script
│   ├── train_qlora.py                  # Model training CLI script
│   ├── predict.py                      # Test inference CLI script
│   ├── evaluate.py                     # Metric evaluation CLI script
│   ├── verify_artifact.py              # SHA-256 verification CLI script
│   └── make_15_commits.sh              # 15 staged commits execution script
├── docs/
│   ├── certificate.png                 # AI Seekho Day 2026 1st Place Certificate
│   ├── methodology.md                  # Detailed architectural report
│   ├── experiments.md                  # Experimental progression across model runs
│   ├── data_audit.md                   # Audit pipeline & contrastive curriculum design
│   ├── reproducibility.md              # Hardware & determinism guide
│   ├── model_card.md                   # Hugging Face model card documentation
│   ├── project_archaeology.md          # Pre-hackathon archive & provenance report
│   └── experiment_lineage.md           # Master experiment tracking table
├── tests/
│   ├── test_cleaning.py                # Cleaning & alias canonicalization tests
│   ├── test_prompt.py                  # Prompt formatting tests
│   ├── test_parser.py                  # Output parser tests
│   └── test_submission.py              # Submission CSV assertion tests
└── outputs/
    └── .gitkeep                        # Output directory for predictions & CSVs
```

---

## 📄 License & Citation

This project is released under the [Apache 2.0 License](LICENSE). Base model weights are subject to Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

```bibtex
@misc{farooq2026gemma4claimverification,
  author = {Farooq, Muhammad Umar},
  title = {Reliable Evidence-Based Claim Verification with Gemma 4 12B},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/omerfarooq223/gemma-claim-verification}}
}
```
