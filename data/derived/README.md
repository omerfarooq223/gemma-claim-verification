# Audited & Derived Datasets

This directory houses cleaned, audited, and contrastive splits derived during system development:

- `train_clean_v3_semantic.jsonl` (935 rows, SHA-256: `b706dbf0c0b4aab4cbcd07bb89c5d018f7c41e47a55b757016ef3d07a9713337`)
- `d4_contrastive_train_v1.jsonl` (150 rows, SHA-256: `17258cbc0e40f4ebd1cd4d583e3a331e59217d62cf4ff36741e8b1e3a7a98f41`)
- `D4_FROZEN_HOLDOUT_V1_DO_NOT_TRAIN.jsonl` (75 rows, SHA-256: `61f41a537f738ecd153474565c1e22bb678c4f8a9dd096ed9eabfe5bddc2e1b2`)
- `external_stress_30_v1.jsonl` (30 rows, SHA-256: `05624200845a79228ac4e05540b013ea0836d049802fcd3f9dcc60ada8aecedc`)

To generate `train_clean_v3_semantic.jsonl` from raw data, use `scripts/clean_training_data.py`.
