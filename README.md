# Gemma Claim Verification

I built this project for **AI Seekho Day 2026**, where it placed **first**. It uses a QLoRA adapter on Gemma 4 12B to classify a claim against supplied evidence. The selected checkpoint scored **94.40% accuracy** and **94.38% macro-F1** on the 500-example event-day test set.

[Model adapter](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora) · [Live demo](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier) · [Source code](https://github.com/omerfarooq223/gemma-claim-verification)

<p align="center">
  <a href="docs/certificate.png">
    <img src="docs/certificate.png" alt="First-place certificate for the Gemma Fine-Tuning Competition at AI Seekho Day 2026" width="820">
  </a>
</p>

<p align="center"><em>First place in the Gemma Fine-Tuning Competition at AI Seekho Day 2026.</em></p>

## What it does

Give it a claim and one or more evidence passages. It returns one of three labels:

| Label | Meaning |
|---|---|
| `SUPPORTS` | The evidence establishes the claim. |
| `REFUTES` | The evidence contradicts the claim. |
| `NOT_ENOUGH_INFO` | The evidence does not settle the claim either way. |

For example:

```text
Claim: The company's revenue declined in 2023.
Evidence: Revenue rose from $1.2 billion in 2022 to $1.5 billion in 2023.

FINAL: REFUTES
```

The model judges the evidence you provide. It does not search the web or check whether the source itself is reliable. Evidence-only prompting is part of the training setup, not a guarantee that every prediction is correct.

## Results

These are the recorded results for the selected competition checkpoint, not fresh evaluations of the hosted demo.

| Evaluation set | Examples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Organizer validation | 300 | 92.33% | 92.27% |
| External stress set | 30 | 93.33% | 93.64% |
| Blind holdout | 120 | 93.33% | 93.37% |
| Event-day test | 500 | **94.40%** | **94.38%** |

On the event-day test, 472 of 500 predictions were correct, with no invalid outputs. Most errors were contradictions classified as support: 26 examples with a true label of `REFUTES` were predicted as `SUPPORTS`.

The [experiment history](docs/experiments.md) includes the other evaluation sets, earlier checkpoints, and the reproduction run.

## Run it

Use Python 3.10 or newer and an NVIDIA CUDA GPU for the documented 4-bit setup. The reference training environment used a T4 with 16 GB of VRAM. CPU-only and Apple Silicon inference are not validated here.

```bash
git clone https://github.com/omerfarooq223/gemma-claim-verification.git
cd gemma-claim-verification
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[app]'
python app.py
```

Open `http://localhost:7860`. The first model load downloads the base weights and adapter, so it takes longer than later requests.

The published weights are a **LoRA adapter**, not a standalone model. The app loads both:

- Base: [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it)
- Adapter: [`omerfarooq223/gemma-4-12b-evidence-verification-qlora`](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora)

If Hugging Face requests authentication, run `hf auth login` locally. On Spaces, use an `HF_TOKEN` secret in the Space settings. See the [deployment guide](HUGGINGFACE_GUIDE.md) for the Space configuration and troubleshooting.

### Predict from a file

After installing the package, run:

```bash
python scripts/predict.py \
  --test data/examples/sample.jsonl \
  --adapter omerfarooq223/gemma-4-12b-evidence-verification-qlora \
  --config configs/final_inference.yaml \
  --output outputs/submission.csv
```

You can also pass a local adapter directory to `--adapter`. Input is JSONL, with one record per line:

```json
{"id": "example-1", "claim": "Revenue declined in 2023.", "evidence": ["Revenue rose from $1.2B in 2022 to $1.5B in 2023."], "label": "REFUTES"}
```

The `label` field is optional for prediction and required for evaluation. The included sample is a smoke test, not a benchmark.

```bash
python scripts/evaluate.py \
  --gold data/examples/sample.jsonl \
  --predictions outputs/submission.csv \
  --output_json outputs/metrics.json
```

## How I trained it

The main work was cleaning the data and targeting the mistakes the model kept making. The original training set had 1,000 rows; the audited set contained 935. I added 150 contrastive examples covering numerical changes, swapped entities, and incomplete evidence, bringing the final training set to **1,085 examples**.

The contrastive pool contained 225 examples arranged in groups of three, with shared evidence and a different claim for each label. Of those examples, 150 went into training and 75 were held out.

| Setting | Value |
|---|---|
| Base model | Gemma 4 12B instruction-tuned |
| Quantization | 4-bit NF4, double quantization, FP16 compute |
| LoRA | Rank 8, alpha 16, dropout 0.05 |
| Target projections | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `2e-4` |
| Effective batch size | 16 |
| Training | 2 epochs, 136 optimizer steps |
| Loss | Assistant completion only: `FINAL: <LABEL>` |
| Inference | Greedy decoding, thinking disabled, up to 24 new tokens |

The final adapter was trained from a fresh base model. I kept that checkpoint after a later reproduction run improved validation accuracy but performed worse on the external stress set.

For the full process, see the [data audit](docs/data_audit.md), [methodology](docs/methodology.md), and [final competition notebook](notebooks/final_competition_notebook.ipynb).

### Retraining

The competition data and audited contrastive files are not included in a fresh clone. Read [data/README.md](data/README.md) for the expected files. The cleaning script handles structural cleanup; reproducing the selected training set also requires the documented semantic audit and recovered data.

Once those files are available:

```bash
python scripts/train_qlora.py \
  --config configs/final_train.yaml \
  --train_clean data/derived/train_clean_v3_semantic.jsonl \
  --contrastive data/derived/d4_contrastive_train_v1.jsonl \
  --output_dir checkpoints/final_adapter
```

Retraining is not guaranteed to produce identical weights. To reproduce the recorded checkpoint results, use the selected adapter and verify its checksum:

```bash
python scripts/verify_artifact.py \
  checkpoints/final_adapter/adapter_model.safetensors \
  --key selected_final_adapter
```

Expected SHA-256:

```text
76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e
```

Additional hashes and environment details are in the [reproducibility guide](docs/reproducibility.md).

## Repository guide

| Path | Contents |
|---|---|
| `app.py` | Gradio demo for local use and Hugging Face Spaces |
| `src/gemma_claim_verification/` | Data cleaning, prompts, model loading, training, inference, and evaluation |
| `scripts/` | Command-line entry points |
| `configs/` | Training and inference settings |
| `tests/` | Tests for data processing and inference behavior |
| `notebooks/` | Final competition run and development history |
| `docs/` | Methodology, experiments, model card, and provenance |
| `data/` | Sample records and instructions for the competition files |

Run the tests without downloading model weights:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Author and license

Built by [Muhammad Umar Farooq](https://github.com/omerfarooq223).

The repository is licensed under [Apache 2.0](LICENSE). Refer to the [base model card](https://huggingface.co/google/gemma-4-12B-it) for the base model's terms and documentation.
