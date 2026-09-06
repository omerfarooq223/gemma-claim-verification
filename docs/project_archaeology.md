# Project Archaeology & Provenance Report

This document records the historical provenance, pre-hackathon exploration, and workspace artifacts uncovered during the comprehensive forensic audit.

---

## 1. Important Naming Note

> [!IMPORTANT]
> **Pre-Hackathon Rehearsal vs Official A-Series Experiments**:
> Pre-hackathon rehearsal materials referred informally to “Dataset A” and “Dataset B”. **These should not be confused with the later A1/A2/A3 experimental series conducted after receipt of the official organizer dataset.**
>
> - **Pre-hackathon "Dataset A"**: An early rehearsal/planning baseline used to test pipeline mechanics prior to competition data release.
> - **Official `A1` / `A2` / `A3`**: Empirical development experiments on the official organizer dataset (using the 795 real fit / 140 internal dev split).

---

## 2. Project Stages & Historical Archaeology

### Stage 0: Pre-Hackathon Rehearsal (Aug 20–25, 2026)
Prior to receiving the official competition data for AI Seekho Day 2026, the team rehearsed fine-tuning workflows on proxy tasks:
- **PakAssist Intent Classifier (`LLM Fine Tunning/`)**: A 60-label bilingual English-Urdu intent routing system fine-tuned on Gemma 4 E2B using QLoRA. This served as general technical rehearsal on a separate task, useful for learning infrastructure, GPU memory allocation, and regex decoding workflows.
- **Dataset A Rehearsal (`Gemma_Hackathon_Team_Rehearsal_Guide_Dataset_A.docx`)**: Pre-competition rehearsal and planning material.
- **Playbook (`Gemma_Fine_Tuning_Hackathon_Playbook.pdf`)**: Preparation and decision framework.

### Stage 1: Gemma 2 2B Baseline & Synthetic Scaling Pitfalls (Aug 26, 2026)
- **A1**: Cleaned real baseline on Gemma 2 2B achieving 82.07% Macro-F1 on organizer validation.
- **A2**: Attempted indiscriminate synthetic expansion (+180 examples), which degraded validation Macro-F1 to 79.28%.
- **A3**: Evaluated prompt refinements (81.73% Macro-F1). Closed the 2B phase with A1 retained as the 2B benchmark.

### Stage 2: Gemma 4 E4B Scaling & Gemma 2 9B Exploration (Aug 27, 2026)
- **B1**: Scaled to Gemma 4 E4B IT, improving validation Macro-F1 to 88.64%.
- **B2**: Re-tested synthetic augmentation (+180 examples) on E4B; Macro-F1 dropped to 85.23%, confirming that uncurated synthetic scaling consistently harmed generalization.
- **Gemma 2 9B**: Explored as an alternative fallback path (`gemma2_9b_factcheck_frozen_model.zip`).

### Stage 3: Gemma 4 12B & Failure-Driven Contrastive Curriculum (Aug 27, 2026)
- **D1 Champion**: Gemma 4 12B IT trained on clean 795 real examples (`2c64a87e...`), achieving 88.00% validation accuracy and 90.00% on External30.
- **D4 Champion**: Constructed 225 contrastive trios, re-audited with a phrase-aware revenue auditor, and trained on 795 real + 150 contrastive examples (`03b134b9...`). Validation accuracy jumped to **92.00%** and Frozen Holdout75 jumped to **86.67%**.

### Stage 4: Full Production Model & Winning Result (Aug 28, 2026)
- **FINAL Model**: Applied the frozen D4 recipe to all 935 clean real examples + 150 contrastive examples (1,085 total examples) starting from a **fresh Gemma 4 12B base and fresh LoRA initialization**.
- **Frozen Adapter**: SHA-256 `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e`.
- **Event-Day Supervised Test500**: **94.40% Accuracy** (472/500), **94.38% Macro-F1**, winning the hackathon.

---

## 3. Checkpoint Provenance Mapping

| Checkpoint Family | Role in Project | LoRA Adapter SHA-256 | Disposition in Public Repo |
|---|---|---|---|
| **FINAL_ALL935_PLUS150** | **Selected Winning System** | `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e` | Canonical checkpoint for production & public release. |
| **D4 Champion** | Finalist contrastive proof | `03b134b92e46f44b3802bf563668436250078a2ffef175e35c0c3c5b5ccf42d7` | Documented in experiment history. |
| **D1 Champion** | Intermediate 12B baseline | `2c64a87e227e0638be9dc73e021f877f2536876ccf94cc116e04d60075164f8f` | Documented in experiment history. |
| **B1 Champion** | Gemma 4 E4B champion | `100f2e9640a586632f7450f8bae7a7fb63136c078185e4f16b665e3d5320213d` | Documented in experiment history. |
| **A1 Champion** | Gemma 2 2B champion | `b975567893982e0cf445d4b1df87f52f99972c49f6058e5e8172926663c4a01e` | Documented in experiment history. |
