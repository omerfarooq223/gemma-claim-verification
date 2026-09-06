# 🤗 Hugging Face Deployment Guide
## Gemma 4 12B Evidence-Based Claim Verification (1st Place AI Seekho Day 2026)
**Author**: Muhammad Umar Farooq

This guide provides comprehensive instructions on **what to deploy** and **how to deploy** your fine-tuned model weights to the **Hugging Face Model Hub** and your interactive web demo to **Hugging Face Spaces**.

---

## 📌 Overview of Deployment Assets

| Target | Asset Type | Source Files | Hugging Face Destination |
|---|---|---|---|
| **Model Hub** | LoRA Adapter Weights | `FINAL_GEMMA4_12B_ALL935_PLUS150.zip` (`epoch_2_adapter/`) | [`omerfarooq223/gemma-4-12b-evidence-verification-qlora`](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora) |
| **Spaces Hub** | Gradio Web App | `app.py`, `requirements.txt`, `README.md` | [`omerfarooq223/gemma-claim-verifier`](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier) (Gradio Space) |

---

## Part 1: Deploying Model Weights to Hugging Face Model Hub

### Step 1.1: Extract Adapter Files
The winning fine-tuned LoRA adapter files are stored in `FINAL_GEMMA4_12B_ALL935_PLUS150.zip`. Extract the adapter folder:

```bash
unzip -q FINAL_GEMMA4_12B_ALL935_PLUS150.zip -d extracted_adapter/
cd extracted_adapter/FINAL_GEMMA4_12B_ALL935_PLUS150/epoch_2_adapter
```

Verify that the directory contains:
- `adapter_model.safetensors` (~131 MB)
- `adapter_config.json`
- `tokenizer.json` (~32 MB)
- `tokenizer_config.json`
- `processor_config.json`
- `chat_template.jinja`
- `README.md`

### Step 1.2: Authenticate with Hugging Face CLI
Make sure you have `huggingface_hub` installed and log in using your Hugging Face Access Token (Write permission):

```bash
pip install huggingface_hub
huggingface-cli login
```

### Step 1.3: Upload Adapter Model to HF Hub
You can upload the adapter directory directly using Python or the CLI:

#### Option A: Using Python (Recommended)
Create and run a small upload script:

```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "omerfarooq223/gemma-4-12b-evidence-verification-qlora"

# Create repository if it doesn't exist
api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

# Upload all files from the extracted adapter folder
api.upload_folder(
    folder_path="FINAL_GEMMA4_12B_ALL935_PLUS150/epoch_2_adapter",
    repo_id=repo_id,
    repo_type="model",
)
print("✅ LoRA Adapter uploaded successfully to Hugging Face!")
```

#### Option B: Using CLI
```bash
huggingface-cli upload omerfarooq223/gemma-4-12b-evidence-verification-qlora . / --repo-type=model
```

---

## Part 2: Deploying Interactive Web App to Hugging Face Spaces

Hugging Face Spaces allows you to host an interactive Gradio web application for free or with GPU acceleration.

### Step 2.1: Create a New Space on Hugging Face
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Set **Owner**: `omerfarooq223`.
3. Set **Space Name**: `gemma-claim-verifier`.
4. Select **SDK**: `Gradio`.
5. Select **Hardware**: CPU Basic (Free) or GPU (T4 / A10G for fast 4-bit NF4 inference).
6. Set **Privacy**: `Public`.

### Step 2.2: Prepare Space Repository Files
Your Space requires the following files:

1. **`app.py`**: The Gradio application script (already generated in root directory).
2. **`requirements.txt`**: Space Python dependencies:
   ```text
   transformers>=4.44.0
   peft>=0.12.0
   bitsandbytes>=0.43.0
   accelerate>=0.33.0
   torch>=2.2.0
   gradio>=4.0.0
   ```
3. **`README.md`**: Space configuration header:
   ```yaml
   ---
   title: Gemma 4 12B Claim Verification Demo
   emoji: 🏆
   colorFrom: blue
   colorTo: indigo
   sdk: gradio
   sdk_version: 4.44.0
   app_file: app.py
   pinned: false
   license: apache-2.0
   short_description: 1st Place Solution AI Seekho Day 2026 Claim Verification
   ---
   ```

### Step 2.3: Push Code to Hugging Face Space
Clone your new Space repository and copy `app.py` and `requirements.txt`:

```bash
git clone https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier hf_space
cp app.py hf_space/
cp requirements.txt hf_space/

cd hf_space
git add app.py requirements.txt
git commit -m "feat: initial Gradio app deployment for Gemma 4 claim verification"
git push
```

Your web application will build automatically and be accessible live at:
`https://huggingface.co/spaces/omerfarooq223/gemma-4-claim-verifier-demo`

---

## Part 3: Programmatic Usage Example

Anyone can now use your published model in Python with 3 lines of code:

```python
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 1. Quantized Base Model
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

# 2. Attach Winning Adapter
model = PeftModel.from_pretrained(
    base_model,
    "omerfarooq223/gemma-4-12b-evidence-verification-qlora",
)

print("🏆 Gemma 4 12B Winning Claim Verifier loaded successfully!")
```
