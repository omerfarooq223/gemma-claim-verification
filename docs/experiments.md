# Experimentation History & Progression

This document chronicles the experimental trajectory of the project, documenting baseline explorations, ablation studies, and the final model selection process.

---

## 1. Experimental Progression Overview

The project progressed across 4 major phases:
1. **Phase 1: Gemma 2 2B Exploration (`A1` – `A3`)**: Baseline establishment and first synthetic augmentation attempts.
2. **Phase 2: Gemma 4 E4B & Gemma 2 9B Exploration (`B1` – `B3`)**: Model scaling and learning rate sensitivity.
3. **Phase 3: Gemma 4 12B & Contrastive Curriculum (`D1` – `D4`)**: Failure analysis, contrastive trios, and validation jumping to $>91\%$.
4. **Phase 4: Full Production Training (`FINAL`)**: Training on all 935 clean real + 150 contrastive examples.

---

## 2. Checkpoint Lineage & Architecture Evolution

| Checkpoint | Training Data Composition | LoRA Adapter SHA-256 | Role & Significance |
|---|---|---|---|
| **D1 Champion** | 795 real fit subset | `2c64a87e227e0638be9dc73e021f877f2536876ccf94cc116e04d60075164f8f` | Gemma 4 12B baseline using the 795-row fit portion (with 140 held as internal dev). |
| **D4 Champion** | 795 real + 150 contrastive = 945 | `03b134b92e46f44b3802bf563668436250078a2ffef175e35c0c3c5b5ccf42d7` | Controlled contrastive experiment establishing the successful recipe (+9.33% jump on frozen holdout). |
| **FINAL Selected** | **935 real + 150 contrastive = 1,085** | `76630ec4620ff7244f3b6c9ef0350617939d33a5bc6f0e9c545816175b646d8e` | **Fresh production training run; official winning checkpoint (94.40% Test500).** |
| **Reproduction** | 935 real + 150 contrastive = 1,085 | `cf24e9701752c7dd6d1f35fc9ceaf933f3cdc2dc8e31b5823ba98c444f6b51a9` | Independent nominal reproduction; higher single validation score but rejected for weaker stress generalization. |

> [!IMPORTANT]
> **Fresh Base Initialization**: The FINAL model was trained from a fresh Gemma 4 12B base with a fresh LoRA initialization. It was not continued from D4. D4 established the successful recipe; the final run then applied that recipe to all 935 audited real examples plus the same 150 audited contrastive examples.

---

## 3. Quantitative Results Across Model Generations

| Experiment ID | Base Model | Training Data | Adapter SHA-256 (Prefix) | Organizer Val (Acc / F1) | External Stress30 (Acc / F1) | Frozen Holdout75 (Acc / F1) | Old Synthetic180 (Acc / F1) | Blind120 (Acc / F1) | Event Test500 (Acc / F1) |
|---|---|---|---|---|---|---|---|---|---|
| **A1 Baseline** | Gemma 2 2B IT | Clean 795 | `b9755678...` | 82.00% / 82.07% | 80.00% / 80.25% | — | 72.22% / 71.80% | — | — |
| **A2 Synthetic** | Gemma 2 2B IT | 795 + 180 Synth | — | 79.33% / 79.28% | 76.67% / 76.40% | — | 74.44% / 73.90% | — | — |
| **A3 Prompt V2** | Gemma 2 2B IT | Clean 795 | — | 81.67% / 81.73% | 80.00% / 80.10% | — | 72.78% / 72.10% | — | — |
| **B1 E4B** | Gemma 4 E4B IT | Clean 795 | `100f2e96...` | 88.67% / 88.64% | 86.67% / 86.80% | — | 76.11% / 75.40% | — | — |
| **B2 E4B Synth** | Gemma 4 E4B IT | 795 + 180 Synth | `5fb01b61...` | 85.33% / 85.23% | 83.33% / 83.10% | — | 78.33% / 77.90% | — | — |
| **D1 Champion** | Gemma 4 12B IT | Clean 795 | `2c64a87e...` | 88.00% / 87.80% | 90.00% / 90.48% | 77.33% / 77.50% | 81.67% / 81.66% | 93.33% / 93.28% | — |
| **D4 Champion** | Gemma 4 12B IT | 795 + 150 Contrastive | `03b134b9...` | 92.00% / 91.94% | 93.33% / 93.64% | **86.67% / 86.58%** | 79.44% / 78.66% | 89.17% / 89.20% | — |
| **FINAL Selected** | **Gemma 4 12B IT** | **935 + 150 Contrastive** | `76630ec4...` | **92.33% / 92.27%** | **93.33% / 93.64%** | **82.67% / 81.79%** | **81.67% / 80.84%** | **93.33% / 93.37%** | **94.40% / 94.38%** |
| **Reproduction Run** | Gemma 4 12B IT | 935 + 150 Contrastive | `cf24e970...` | 93.33% / 93.32% | 86.67% / 86.50% | 81.33% / 80.50% | 78.33% / 77.90% | 92.50% / 92.40% | — |

---

## 4. Key Research Insights

### 1. Data Cleaning Materially Outweighed Model Scaling
Auditing the 1,000 raw training examples, canonicalizing labels, and removing duplicate conflicts produced immediate accuracy improvements. Training larger models on uncleaned data failed to resolve fundamental boundary ambiguities.

### 2. Indiscriminate Synthetic Augmentation Harmed Generalization
In experiments `A2` and `B2`, adding 180 indiscriminately generated synthetic items reduced organizer validation Macro-F1 by **2.79%** on Gemma 2 2B and **3.41%** on Gemma 4 E4B. Synthetic expansion without strict semantic auditing contaminated subtle decision boundaries.

### 3. Targeted Contrastive Trios Solved Specific Boundary Failures
In contrast to broad synthetic scaling, creating 150 audited contrastive trios (testing numerical threshold inversions, partial evidence, and entity contradictions) enabled the jump from `D1` (88.00% Val Acc, 77.33% Holdout Acc) to `D4` (**92.00% Val Acc, 86.67% Holdout Acc**).

### 4. Robustness-Driven Checkpoint Selection vs Single Validation Scores
A later independent run using the nominally identical final training recipe produced a different adapter (`cf24e970...`) and improved organizer-validation accuracy from **92.33% to 93.33%**, but performance declined on independent stress evaluations, including External30 from **93.33% to 86.67%** (-6.66%) and Old Synthetic180 from **81.67% to 78.33%** (-3.34%). We therefore retained the previously frozen `76630...` checkpoint based on broader generalization.

---

## 5. Phase 5: Controlled Equal-Compute Research Benchmark (ICIC 2026 Paper)

In our research paper *"Failure-Targeted Contrastive Augmentation for Low-Resource Claim Verification"* (accepted at the 7th International Conference on Innovative Computing, **ICIC 2026 / IEEE**, Paper ID 85), we formalized an empirical benchmark to address a fundamental question: **Under an equal computational budget, does failure-targeted contrastive augmentation improve low-resource verification more reliably than repeated real exposure or legacy synthetic augmentation?**

### Equal-Compute Setup
- **Matched Optimizer Steps**: Every condition received exactly 136 optimizer updates (2,176 example exposures with batch size 1 and gradient accumulation 16) to eliminate training compute confounds.
- **Base Model Pinning**: `google/gemma-4-12B-it` pinned to exact commit revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.
- **4 Conditions Evaluated**:
  1. `REAL`: 935 cleaned real examples.
  2. `REPEAT`: 935 real + 150 duplicated real examples (50 per class).
  3. `LEGACY`: 935 real + 150 legacy synthetic examples from A2.
  4. `TARGETED`: 935 real + 150 failure-targeted contrastive trios spanning 10 diagnosed error families.

### Prospective Sealed Test500 Evaluation (3 Seeds: 42, 43, 44)
Predictions were generated and SHA-256 frozen before gold test labels were accessed.

| Condition | Training Examples | Macro-F1 (mean ± SD) | Accuracy (mean ± SD) |
|---|---:|---:|---:|
| `REAL` | 935 | 0.9281 ± 0.0146 | 0.9280 ± 0.0144 |
| `REPEAT` | 1,085 | 0.9427 ± 0.0143 | 0.9427 ± 0.0140 |
| `LEGACY` | 1,085 | 0.9362 ± 0.0251 | 0.9360 ± 0.0251 |
| `TARGETED` | 1,085 | **0.9467 ± 0.0117** | **0.9467 ± 0.0117** |

- **Hierarchical Paired-Bootstrap Contrast (10,000 resamples)**:
  - `TARGETED - LEGACY`: +0.0105 [95% CI: -0.0094, +0.0343]
  - `TARGETED - REPEAT`: +0.0040 [95% CI: -0.0183, +0.0260]
  - `TARGETED - REAL`: +0.0186 [95% CI: -0.0025, +0.0383]

### Post-Review Exploratory Extension (5 Seeds: 42–46, 30 Models)
- **`NOTRIO` Breakthrough**: Allocating 150 synthetic examples across the 10 failure families with independent evidence passages achieved **0.9621 ± 0.0162 Macro-F1** on Test500, outperforming `TARGETED` across all five seeds (+0.0177 difference, 95% CI: [+0.0039, +0.0325]).
- **Cross-Domain Transfer**: On SciFact gold-rationale development pairs, `NOTRIO` also led transfer with **0.8332 two-class Macro-F1**.
- **Prompt-Only Baselines**: Pinned Gemma 4 12B IT reached 0.7903 (0-shot) and 0.8252 (3-shot), demonstrating the vital importance of fine-tuning over in-context prompting.
- **Frozen Artifacts**: All tables, figures, and manifests are available in `experiments/paper_v1/paper_assets_v1/`.
