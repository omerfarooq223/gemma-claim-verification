---
library_name: peft
license: apache-2.0
base_model: google/gemma-4-12B-it
tags:
  - gemma4
  - peft
  - lora
  - qlora
  - claim-verification
  - fact-checking
  - evidence-based-reasoning
  - text-classification
pipeline_tag: text-generation
language:
  - en
---

# 🏆 Gemma 4 12B Evidence Verification QLoRA
### **1st Place Winning Solution — AI Seekho Day 2026**

## Model Details

### Model Description

- **Developed by**: Muhammad Umar Farooq
- **Shared by**: Muhammad Umar Farooq
- **Model type**: Fine-tuned PEFT/QLoRA Parameter-Efficient Adapter
- **Language(s) (NLP)**: English (`en`)
- **License**: Apache 2.0
- **Finetuned from model**: [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)

### Model Sources

- **Repository**: [https://github.com/omerfarooq223/gemma-claim-verification](https://github.com/omerfarooq223/gemma-claim-verification)
- **Demo Space**: [https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier)

---

## Uses

### Direct Use
Given a natural language claim and one or more supplied evidence passages, classify the evidence relation into exactly one of three categories:
- `SUPPORTS`
- `REFUTES`
- `NOT_ENOUGH_INFO`

### Downstream Use
Fact-checking applications, automated RAG verification systems, hallucination detection engines, and evidence audit tools.

### Out-of-Scope Use
General factual QA without evidence passages. The model is calibrated strictly to evaluate truth conditions **only relative to the supplied text passages**.

---

## Risk, Bias, and Limitations

- **Bounded Context**: Prompts and evidence passages must fit within 256 tokens.
- **Evidence-Only Scope**: The model does not query external knowledge bases or search the web; it evaluates factual claims strictly grounded in the supplied context.
- **Error Pattern Analysis**: Achieved 99.30% precision on `REFUTES` and 100% recall on `NOT_ENOUGH_INFO`. Subtle numerical or scope contradictions with supportive phrasing can occasionally be mispredicted as `SUPPORTS`.

---

## How to Get Started with the Model

```python
import torch
from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# 1. Load 4-bit NF4 quantized base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained("google/gemma-4-12B-it")
base_model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "google/gemma-4-12B-it",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. Attach Winning LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    "omerfarooq223/gemma-4-12b-evidence-verification-qlora",
)
```

---

## Training Details

### Training Data
- **Audited Clean Base**: 935 semantically audited examples (cleaned from 1,000 raw noisy rows via a 10-step audit pipeline).
- **Audited Contrastive Curriculum**: 150 contrastive examples arranged in trios.
- **Total Training Dataset**: 1,085 curated examples.

### Training Hyperparameters
- **Quantization**: 4-bit NF4 (`bnb_4bit_use_double_quant=True`)
- **Compute Precision**: FP16 (`torch.float16`)
- **LoRA Config**: Rank $r=8$, Alpha $\alpha=16$, Dropout $0.05$
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Trainable Parameters**: 32,784,384 (~0.27% of base model)
- **Optimizer**: AdamW (`lr=2e-4`, `weight_decay=0.01`)
- **Effective Batch Size**: 16 (Batch size 1, Gradient accumulation 16)
- **Epochs**: 2 (136 total optimizer steps)
- **Loss Masking**: Completion-only loss (`FINAL: <LABEL>`)

---

## Evaluation Benchmark Performance

### Supervised Event-Day Test Set (500 Examples)

$$\text{Overall Accuracy}: \mathbf{94.40\%} \quad (472 / 500 \text{ correct}) \qquad \text{Macro-F1}: \mathbf{94.38\%}$$

| Label Class | Precision | Recall | F1-Score | Support (Ground Truth) |
|---|---|---|---|---|
| **`SUPPORTS`** | 86.39% | 98.80% | **92.18%** | 167 |
| **`REFUTES`** | 99.30% | 84.43% | **91.26%** | 167 |
| **`NOT_ENOUGH_INFO`** | 99.40% | 100.00% | **99.70%** | 166 |
| **Macro Average** | **95.03%** | **94.41%** | **94.38%** | **500** |

---

## Citation

```bibtex
@misc{farooq2026gemma4claimverification,
  author = {Farooq, Muhammad Umar},
  title = {Reliable Evidence-Based Claim Verification with Gemma 4 12B},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora}}
}
```
