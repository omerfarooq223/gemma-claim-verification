# Complete Experiment Lineage

This table catalogues every experiment conducted during the project lifecycle, tracking training configuration, evaluation results, and disposition status.

| Experiment ID | Period | Dataset | Base Model | Key Configuration | Validation F1 / Acc | External Stress F1 / Acc | Status |
|---|---|---|---|---|---|---|---|
| **E0 Baseline** | Pre-Hackathon | Sample Data | Gemma 2 2B IT | Default Colab Starter | ~80.0% / ~80.0% | — | Pre-hackathon starter |
| **PakAssist E3** | Pre-Hackathon | Bilingual 60-Class | Gemma 4 E2B | QLoRA $r=16$, lr=2e-4 | 76.65% Macro-F1 | — | Pre-hackathon rehearsal |
| **A1 Baseline** | Aug 26 | Clean V2 (795 fit) | Gemma 2 2B IT | QLoRA $r=16$, lr=1e-4, 2 ep | 82.07% / 82.00% | 80.25% / 80.00% | 2B Champion (Closed) |
| **A2 Synth Scaling** | Aug 26 | 795 + 180 Synth | Gemma 2 2B IT | QLoRA $r=16$, lr=1e-4, 2 ep | 79.28% / 79.33% | 76.40% / 76.67% | Failed (Degraded) |
| **A3 Prompt V2** | Aug 26 | Clean V2 (795 fit) | Gemma 2 2B IT | Prompt V2 wording | 81.73% / 81.67% | 80.10% / 80.00% | Superseded |
| **B1 E4B Baseline** | Aug 27 | Clean 795 | Gemma 4 E4B IT | QLoRA $r=8$, lr=2e-4, 2 ep | 88.64% / 88.67% | 86.80% / 86.67% | E4B Champion |
| **B2 E4B Synth** | Aug 27 | 795 + 180 Synth | Gemma 4 E4B IT | QLoRA $r=8$, lr=2e-4, 2 ep | 85.23% / 85.33% | 83.10% / 83.33% | Failed (Degraded) |
| **B3 E4B lr=1e-4** | Aug 27 | Clean 795 | Gemma 4 E4B IT | lr=1e-4 challenger | 86.93% / 87.00% | 85.50% / 85.67% | Superseded |
| **Gemma 2 9B** | Aug 27 | Clean 795 | Gemma 2 9B IT | QLoRA $r=8$, lr=2e-4, 2 ep | ~88.5% / ~88.5% | — | Fallback Exploration |
| **D1 Champion** | Aug 27 | Clean V3 (795 fit) | Gemma 4 12B IT | QLoRA $r=8$, lr=2e-4, 2 ep | 87.80% / 88.00% | 90.48% / 90.00% | Intermediate Champion |
| **D4 Champion** | Aug 27 | 795 + 150 Contrastive | Gemma 4 12B IT | QLoRA $r=8$, lr=2e-4, 2 ep | 91.94% / 92.00% | 93.64% / 93.33% | Finalist Champion |
| **FINAL Selected** | **Aug 28** | **935 + 150 Contrastive** | **Gemma 4 12B IT** | **QLoRA $r=8$, lr=2e-4, 2 ep** | **92.27% / 92.33%** | **93.64% / 93.33%** | **SELECTED WINNING SYSTEM** (94.40% Test500) |
| **Reproduction Run** | Aug 28 | 935 + 150 Contrastive | Gemma 4 12B IT | Retrained replicate | 93.32% / 93.33% | 86.50% / 86.67% | Rejected (Poor generalization) |
