# Competition & Development Notebooks

This directory preserves both the historical master development trajectory and the clean submission notebook used during the hackathon.

---

## Notebook Artifacts

### 1. [`final_competition_notebook.ipynb`](final_competition_notebook.ipynb) (Canonical Submission Notebook)
The self-contained, clean Kaggle submission notebook that:
- Executes data integrity and SHA-256 preflight checks.
- Demonstrates the 10-step data cleaning pipeline.
- Loads the frozen winning Gemma 4 12B QLoRA adapter (`76630ec4...`).
- Runs deterministic inference on the 500-row test benchmark.
- Generates the winning `submission.csv`.

### 2. [`full_development_history_notebook.ipynb`](full_development_history_notebook.ipynb) (Master A-to-Z Trajectory)
The complete master development notebook preserving all 47 historical steps:
- **Steps 1–16 (A-Series)**: Gemma 2 2B baselines, data audit, and synthetic scaling tests.
- **Steps B0–B3 (B-Series)**: Gemma 4 E4B scaling and learning rate challengers.
- **Steps D1–D4 (D-Series)**: Gemma 4 12B exploration, contrastive trios synthesis, and revenue auditor direction correction.
- **Cell 7 (FINAL Production Run)**: Exact training log on all 1,085 examples (935 real + 150 contrastive) yielding adapter `76630ec4...`.

---

> [!NOTE]
> For production use, the modular package in `src/gemma_claim_verification/` and CLI tools in `scripts/` provide the maintainable implementation.
