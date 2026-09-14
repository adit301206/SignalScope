# SignalScope Model Report

## 1. Model Overview

SignalScope uses a transfer-learning based image classifier to distinguish between:

- **REAL** — likely captured/photographic imagery
- **AI-GENERATED** — likely synthetically generated imagery

The final classifier is based on **EfficientNet-B0 pretrained on ImageNet** and fine-tuned for binary classification.

The model outputs:

1. Predicted class
2. AI probability
3. Real probability
4. Confidence score

Grad-CAM is also used as an explanation aid to visualize image regions that contributed more strongly to the model prediction.

---

## 2. Model Architecture

| Property | Configuration |
|---|---|
| Architecture | EfficientNet-B0 |
| Initialization | ImageNet pretrained |
| Task | Binary image classification |
| Number of classes | 2 |
| Classes | REAL, AI-GENERATED |
| Input size | 224 × 224 |
| Framework | PyTorch |
| Final model | `model/best_efficientnet_b0_mixed.pth` |
| Device used for training/evaluation | NVIDIA CUDA GPU |

The final model was trained using a mixed dataset containing CIFAKE and a small number of samples from the Defactify dataset.

---

## 3. Datasets

### 3.1 CIFAKE

CIFAKE was used as the primary training dataset.

Dataset split:

- Training: 100,000 images
- Test: 20,000 images
- Total: 120,000 images

The dataset contains real images from CIFAR-10 and AI-generated images produced using Stable Diffusion v1.4.

For the mixed training experiment:

- 90,000 CIFAKE training images were used for training.
- 10,000 CIFAKE images were retained as the validation set.

---

### 3.2 Defactify

Defactify was introduced to improve generalization beyond the generator distribution represented by CIFAKE.

The dataset contains:

- Real images
- Stable Diffusion 2.1
- Stable Diffusion XL
- Stable Diffusion 3
- DALL-E 3
- Midjourney

For mixed training, 1,000 Defactify training images were added:

| Category | Training Samples |
|---|---:|
| Real | 500 |
| Stable Diffusion 2.1 | 100 |
| Stable Diffusion XL | 100 |
| Stable Diffusion 3 | 100 |
| DALL-E 3 | 100 |
| Midjourney | 100 |
| **Total** | **1,000** |

The Defactify evaluation set was kept separate from training.

---

## 4. Baseline Model

An EfficientNet-B0 model was first trained using CIFAKE.

The baseline was trained for 1 epoch.

### CIFAKE Test Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9973 |
| Macro-F1 | 0.9723 |
| Accuracy | 0.9723 |
| False Positive Rate | 4.32% |
| False Negative Rate | 1.22% |

### Confusion Matrix

```text
                 Predicted
                 REAL   AI
Actual REAL      9568   432
Actual AI         122  9878