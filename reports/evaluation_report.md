# Evaluation Report

## Scope and reproducibility

This report distinguishes recorded experiment outputs from final claims. The raw datasets, manifests, run logs, and mixed-model checkpoint are not tracked, so the results below are reproducible only after supplying equivalent local assets. All binary evaluations threshold the AI softmax probability at 0.5.

## CIFAKE baseline (recorded)

`reports/baseline_results.md` records a one-epoch EfficientNet-B0 baseline on the CIFAKE test split.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9973 |
| Macro-F1 | 0.9723 |
| Accuracy | 0.9723 |
| Confusion matrix | `[[9568, 432], [122, 9878]]` |
| FPR | 0.0432 |
| FNR | 0.0122 |

Rows in the matrix are true real/AI labels and columns are predicted real/AI labels. These are in-distribution CIFAKE results, not unseen-generator results.

## Mixed-model held-out Defactify evaluation (recorded)

The mixed experiment added a selected Defactify training sample to CIFAKE training and evaluated the mixed checkpoint on a separate 600-image Defactify manifest: 100 real, 100 Stable Diffusion 2.1, 100 SDXL, 100 Stable Diffusion 3, 100 DALL-E 3, and 100 Midjourney 6. The tracked `mixed_defactify_results.csv` contains per-image probabilities/predictions; `mixed_results.md` records the aggregate run.

| Metric | CIFAKE-only baseline on Defactify | Mixed model on Defactify |
|---|---:|---:|
| ROC-AUC | 0.6174 | 0.8788 (reported rounded as 0.8790 in `mixed_results.md`) |
| Macro-F1 | 0.4545 | 0.7303 |
| Accuracy | 0.8333 | 0.8200 |
| FPR | 1.0000 | 0.2700 |
| FNR | 0.0000 | 0.1620 |

Mixed-model confusion matrix: `[[73, 27], [81, 419]]`. It corresponds to 73/100 real images correctly classified as real and 419/500 AI images correctly classified as AI. A high accuracy for the CIFAKE-only baseline here is misleading because it predicted all real images as AI; macro-F1, FPR, and ROC-AUC expose that failure.

Recorded AI detection rates for the mixed model are DALL-E 3 96%, Midjourney 6 85%, SDXL 83%, Stable Diffusion 2.1 82%, and Stable Diffusion 3 73%. These generators are represented in the selected Defactify training data, so calling this a strict unseen-generator evaluation for the mixed model would be inaccurate. It is a held-out image split, not a generator-held-out evaluation.

## Robustness (recorded)

`src/robustness/training/test_pipeline.py` applies each transform to a balanced 600-image Defactify sample and writes `reports/robustness/robustness_results.csv`.

| Condition | ROC-AUC | Macro-F1 | Accuracy | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Original | 0.8788 | 0.7303 | 0.8200 | 0.2700 | 0.1620 |
| JPEG quality 50 | 0.8591 | 0.6710 | 0.7500 | 0.2200 | 0.2560 |
| Resize to 112 then 224 | 0.7720 | 0.6052 | 0.8433 | 0.8000 | 0.0280 |
| Gaussian blur radius 1.5 | 0.7973 | 0.6738 | 0.8283 | 0.5800 | 0.0900 |
| Brightness ×1.15 | 0.8820 | 0.7419 | 0.8417 | 0.3400 | 0.1220 |

Brightness having a slightly higher AUC than original is an observed result for this finite sample, not evidence of a general improvement. Resizing and blur raise false positives substantially. No crop, noise, contrast, adversarial attack, or real-time latency test is implemented.

## Generator attribution (recorded, experimental)

The six-class attribution evaluation reports 600 Defactify images (100 per class), test accuracy 0.7400, macro-F1 0.7366, and top-two accuracy 0.9067. Validation values stored in the report are accuracy 0.8440 and macro-F1 0.8445. Per-class correct predictions: real 82/100, Stable Diffusion 2.1 53/100, SDXL 73/100, Stable Diffusion 3 53/100, DALL-E 3 86/100, and Midjourney 6 97/100. The full confusion matrix and class precision/recall/F1 are in [attribution evaluation](attribution/evaluation_30k.txt).

This model is trained on the same generator labels it predicts. The checkpoint is absent and attribution is neither a source-proof mechanism nor a production-ready claim.

## Missing measurements

No calibration curve, expected calibration error, latency, throughput, memory, hardware-independent benchmark, final model selection log, or formal error-analysis artifact is tracked. No result is labelled final in this documentation.
