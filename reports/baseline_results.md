# SignalScope — CIFAKE Baseline

## Model

- Architecture: EfficientNet-B0
- Pretrained: ImageNet
- Input size: 224 × 224
- Batch size: 16
- Learning rate: 0.0001
- Epochs: 1
- Optimizer: AdamW
- Weight decay: 0.0001

## Dataset

- Dataset: CIFAKE
- Training images: 90,000
- Validation images: 10,000
- Test images: 20,000
- Classes:
  - 0 = REAL
  - 1 = FAKE

## CIFAKE Test Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9973 |
| Macro-F1 | 0.9723 |
| Accuracy | 0.9723 |

## Confusion Matrix

```text
[[9568, 432],
 [122, 9878]]