# Reproducibility Guide & Integrity Records

This document details the exact software environment, hardware specifications, determinism considerations, and artifact checksums necessary to reproduce the project results.

---

## 1. Reference Hardware & Environment

The winning model checkpoint was trained and verified in the following reference environment:

- **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
- **CUDA Driver**: CUDA 12.x
- **Python**: Python 3.10 / 3.11
- **PyTorch**: `torch >= 2.2.0`
- **Transformers**: `transformers == 5.10.1`
- **PEFT**: `peft == 0.19.1`
- **BitsAndBytes**: `bitsandbytes == 0.50.1`

---

## 2. Determinism & QLoRA Retraining Realities

> [!IMPORTANT]
> **Why 4-Bit QLoRA Training is Not Bit-for-Bit Deterministic**
> While all random seeds (`seed=42`), data split orders (`seed + epoch`), and hyperparameters were strictly locked, **4-bit low-precision CUDA kernels (NF4 dequantization and non-deterministic atomic reductions) do not guarantee bit-identical floating-point outputs across different runs or GPU architectures**.
>
> In our experimentation:
> - The frozen development checkpoint produced adapter SHA-256 `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e`.
> - An independent rerun with the identical recipe produced adapter SHA-256 `cf24e9701752c7dd6d1f35fc9ceaf933f3cdc2dc8e31b5823ba98c444f6b51a9`.
>
> The project therefore distributes the **exact frozen adapter weights** for production and official benchmark reproduction, while documenting the retraining pipeline for scientific transparency.

---

## 3. Cryptographic Artifact Checksum Registry

| Artifact Key | File / Component | SHA-256 Checksum |
|---|---|---|
| `selected_final_adapter` | `adapter_model.safetensors` (Winning Model) | `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e` |
| `d4_champion_adapter` | `adapter_model.safetensors` (D4 Finalist) | `03b134b92e46f44b3802bf563668436250078a2ffef175e35c0c3c5b5ccf42d7` |
| `d1_champion_adapter` | `adapter_model.safetensors` (D1 Champion) | `2c64a87e227e0638be9dc73e021f877f2536876ccf94cc116e04d60075164f8f` |
| `final_submission_csv` | `submission.csv` (Official Test500 Predictions) | `d4df310da9eb95205daf6bc9ad1f8cc874f7afe5f6d432ce8373f0a9f88dd012` |
| `clean_train935` | `train_clean_v3_semantic_recovered.jsonl` | `b706dbf0c0b4aab4cbcd07bb89c5d018f7c41e47a55b757016ef3d07a9713337` |
| `contrastive_train150` | `d4_contrastive_train_v1.jsonl` | `17258cbc0e40f4ebd1cd4d583e3a331e59217d62cf4ff36741e8b1e3a7a98f41` |
| `frozen_holdout75` | `D4_FROZEN_HOLDOUT_V1_DO_NOT_TRAIN.jsonl` | `61f41a537f738ecd153474565c1e22bb678c4f8a9dd096ed9eabfe5bddc2e1b2` |
| `external_stress30` | `external_stress_30_v1.jsonl` | `05624200845a79228ac4e05540b013ea0836d049802fcd3f9dcc60ada8aecedc` |

---

## 4. Verification Workflow

To verify any local file against the registered hash registry:

```bash
# Verify the final test submission
python scripts/verify_artifact.py outputs/submission.csv --key final_submission_csv

# Verify an adapter weight binary
python scripts/verify_artifact.py checkpoints/final_adapter/adapter_model.safetensors --key selected_final_adapter
```
