# Run the demo on Hugging Face Spaces

This guide covers deploying the interactive claim verification app on Hugging Face Spaces. The [Live Demo](https://huggingface.co/spaces/omerfarooq223/gemma-claim-verifier) loads the [Base Model](https://huggingface.co/google/gemma-4-12B-it) and attaches the [Model Adapter](https://huggingface.co/omerfarooq223/gemma-4-12b-evidence-verification-qlora) from the [Source Repository](https://github.com/omerfarooq223/gemma-claim-verification).

## Space files

Copy the repository's `app.py` and `requirements.txt` into the Space. Keep a separate Space README with this configuration header; the project README is intended for GitHub.

```yaml
---
title: Gemma Claim Verification
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Classify claims against supplied evidence with Gemma 4 12B
---
```

Use ZeroGPU or an NVIDIA GPU with enough memory for the 12B model. CPU Basic is not a validated target for this NF4 setup. On ZeroGPU, the app prepares the model at startup and decorates inference with `spaces.GPU`, following the [ZeroGPU model-loading guidance](https://huggingface.co/docs/hub/spaces-zerogpu#model-loading).

If access requires authentication, add a read token as the `HF_TOKEN` secret in the Space settings. Do not put the token in source files. The token's account must have access to both model repositories.

After uploading the files, wait for the build and model download to complete. Test one example for each label and inspect the runtime logs if inference fails. Dependency resolution alone does not verify GPU execution.

## Errors found in the previous deployment

The public Space files were inspected on September 7, 2026. The Space was sleeping at the time, so these findings come from its source and configuration, not a captured live inference traceback.

| Issue | Correction |
|---|---|
| `AutoModelForCausalLM` was called without being imported. | Load `Gemma4UnifiedForConditionalGeneration`, the architecture specified in the base model config. |
| The unified processor imported Torchvision, but `torchvision` was not installed. | Install the matching `torch==2.8.0` and `torchvision==0.23.0` releases. |
| `huggingface_hub<1.0.0` excluded the Hub version required by Transformers 5.10.1. | Use Hub `>=1.5.0,<2` and the recorded Transformers version. |
| Gradio 4.44 used the older Hub API; the old FastAPI and Starlette caps also prevented upgrading Gradio. | Use Gradio 5.49.1 and remove the conflicting caps. Update `sdk_version` too. |
| The app sent a raw prompt instead of the training chat format. | Apply the processor's chat template with thinking disabled and no duplicate special tokens. |
| Model download and loading happened inside the ZeroGPU request. | Prepare the large base model during ZeroGPU startup. Attach the PEFT adapter inside the first GPU-backed request because Safetensors cannot place it on CUDA before ZeroGPU allocates a real GPU. |

The adapter targets `model.language_model.layers...`, so retaining the unified model wrapper matters. Merely adding the missing CausalLM import does not address that architecture requirement. See the [base model config](https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json).

A `401` or `403` response points to authentication or repository access. An out-of-memory error points to the runtime hardware. Neither can be confirmed from the source alone.

## Adapter files

The model repository needs `adapter_config.json` and `adapter_model.safetensors`. The saved processor, tokenizer, and chat template files should accompany the adapter.

If using the local competition archive, extract it with:

```bash
unzip -q FINAL_GEMMA4_12B_ALL935_PLUS150.zip -d extracted_adapter/
```

Its adapter directory is `extracted_adapter/FINAL_GEMMA4_12B_ALL935_PLUS150/epoch_2_adapter/`. Verify the weights before uploading:

```bash
python scripts/verify_artifact.py \
  extracted_adapter/FINAL_GEMMA4_12B_ALL935_PLUS150/epoch_2_adapter/adapter_model.safetensors \
  --key selected_final_adapter
```

The archive is a local artifact, not part of a fresh Git clone. See [artifacts/README.md](artifacts/README.md) for artifact details.
