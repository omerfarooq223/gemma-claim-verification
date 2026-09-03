# Dataset Documentation

This directory documents the data schemas, cleaning protocol, and curriculum structure for the **Gemma 4 Claim Verification** project.

In compliance with competition guidelines and privacy best practices, **raw competition datasets and private evaluation splits are not tracked in Git**. Instead, this directory provides schema definitions, synthetic sample rows, and reproduction scripts.

---

## 1. Schema Specifications

### Training & Validation Record Format (`.jsonl`)

Each line is an independent JSON object containing:

```json
{
  "id": "example_id_001",
  "claim": "Claim text to be evaluated against evidence.",
  "evidence": [
    "First evidence passage.",
    "Second evidence passage."
  ],
  "label": "SUPPORTS"
}
```

#### Field Descriptions:
- `id` (`str`): Unique identifier for the example.
- `claim` (`str`): The declarative claim statement.
- `evidence` (`List[str]` or `str`): One or more text passages providing context.
- `label` (`str`): Ground-truth classification. Must be one of `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`.

### Test Record Format (`.jsonl`)

Test records follow the same schema with the `label` field omitted:

```json
{
  "id": "test_id_001",
  "claim": "Claim text to be classified.",
  "evidence": [
    "First evidence passage."
  ]
}
```

---

## 2. Dataset Provenance & Integrity Table

| Dataset Split | Examples | SHA-256 Checksum | Class Distribution (S / R / N) | Purpose / Description |
|---|---|---|---|---|
| **Raw Train1000** | 1,000 | `26ea9a6998815d0f99f45aff2206f435781cc71bd92638101d7ea08c2e175d3c` | 365 / 332 / 293 (noisy aliases, 10 blank) | Original organizer training dataset. |
| **Organizer Val300** | 300 | `722a5f111693996369a02d26eda08f00cdfb51f5b8b842addb752814f03716c9` | 100 / 100 / 100 | Primary development validation signal. |
| **Audited Clean935** | 935 | `b706dbf0c0b4aab4cbcd07bb89c5d018f7c41e47a55b757016ef3d07a9713337` | 348 / 311 / 276 | Cleaned real training set after 10-step audit. |
| **Contrastive150** | 150 | `17258cbc0e40f4ebd1cd4d583e3a331e59217d62cf4ff36741e8b1e3a7a98f41` | 50 / 50 / 50 | Targeted contrastive trios for boundary repair. |
| **Total Curriculum** | **1,085** | — | **398 / 361 / 326** | **Final training set for the winning system.** |
| **Event Test500** | 500 | `8811c713e4297c0ce6e89a020d212621af538c74fa3866f4b5d98914ca3b8a28` | Unlabeled | Official event-day evaluation benchmark. |

---

## 3. Directory Layout

```
data/
├── README.md                   # This documentation
├── examples/
│   └── sample.jsonl            # Synthetic examples demonstrating all 3 classes
├── pre_hackathon/
│   └── README.md               # Summary of pre-hackathon rehearsal datasets
├── competition/
│   └── README.md               # Instructions for local organizer datasets
└── derived/
    └── README.md               # Documentation of cleaned & contrastive splits
```

---

## 4. How to Clean Raw Data Locally

To clean and audit a local raw training file:

```bash
python scripts/clean_training_data.py \
    --input data/competition/train.jsonl \
    --output data/derived/train_clean_935.jsonl
```
