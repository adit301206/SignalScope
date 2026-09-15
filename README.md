# SignalScope

### Telling Real From Synthetic in the Age of Generative Media

SignalScope is an end-to-end computer-vision system for assessing whether an image is **likely real or AI-generated**. Instead of treating detection as a binary accusation, SignalScope combines a trained visual detector with explainability, generator attribution, robustness analysis, and provenance signals.

Built for **SIH 2026 — Internal Hackathon**.

---

## ✨ What SignalScope Does

Given a single image, SignalScope can provide:

- **Real vs. AI-generated classification**
- **Confidence / probability scores**
- **Grad-CAM visual explanation** showing image regions that influenced the detector
- **AI generator attribution** across supported generator classes
- **Robustness evaluation** under common image degradations
- **EXIF metadata inspection**
- **C2PA / Content Credentials inspection**
- A web interface for interactive analysis
- A FastAPI backend for model inference

> **Important:** SignalScope provides a likelihood-based assessment. It is not proof of image origin, and provenance/metadata signals should not be interpreted as proof that an image is real or AI-generated.

---

## 🏆 SIH Modules Implemented

| Module | Status | Implementation |
|---|---|---|
| Core: Real vs. AI detection | ✅ | EfficientNet-B0 classifier |
| Bonus A: Explanation | ✅ | Grad-CAM heatmap |
| Bonus B: Generator attribution | ✅ | Multi-class attribution model |
| Bonus C: Robustness | ✅ | JPEG, resize, blur, brightness evaluation |
| Bonus D: Provenance | ✅ | EXIF + C2PA |
| Bonus F: Deployable interface | ✅ | React/Vite frontend + FastAPI backend |
| Bonus E: Image-text consistency | ❌ | Not implemented |
| Bonus G: Active defence | ❌ | Not implemented |

---

# 🧠 System Overview

```text
                         ┌──────────────────────┐
                         │      User Image      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Image Preprocessing │
                         │      224 × 224       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │   EfficientNet-B0 Detector   │
                    │      Real vs AI-generated    │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌────────────┐ ┌────────────┐ ┌───────────────┐
             │   Verdict  │ │  Grad-CAM  │ │  Attribution  │
             │ + confidence│ │  heatmap   │ │  generator    │
             └────────────┘ └────────────┘ └───────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │   Provenance Layer   │
                         │    EXIF + C2PA       │
                         └──────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │   Responsible UI     │
                         └──────────────────────┘
```

The detector uses **EfficientNet-B0 pretrained on ImageNet** and is fine-tuned for binary classification.

---

# 📊 Model & Evaluation

## Final detector

- **Backbone:** EfficientNet-B0
- **Pretrained:** ImageNet
- **Input size:** 224 × 224
- **Classes:**
  - `0` → Real
  - `1` → AI-generated
- **Inference threshold:** 0.5
- **Automatic device selection:** CUDA when available, otherwise CPU

The inference checkpoint included in this repository is:

```text
model/best_efficientnet_b0_mixed.pth
```

Generator attribution uses:

```text
model/generator_attribution_30k.pth
```

---

## Training data

### CIFAKE

CIFAKE contains 120,000 images:

- 60,000 real images derived from CIFAR-10
- 60,000 AI-generated images produced using Stable Diffusion 1.4
- 100,000 training images
- 20,000 test images

For SignalScope, the CIFAKE training split was used as the main training distribution.

Source:

- https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images

CIFAKE is published under the MIT license stated by its dataset source.

### Defactify

Defactify provides real images plus AI-generated images from:

- Stable Diffusion 2.1
- Stable Diffusion XL
- Stable Diffusion 3
- DALL-E 3
- Midjourney 6

The dataset contains 96,000 images overall.

Source:

- https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
- Paper: https://arxiv.org/abs/2601.00553

For SignalScope, Defactify was used to improve cross-generator generalization and for held-out evaluation.

---

# 🔬 Data Split & Generalization Strategy

The important goal was not simply to achieve high accuracy on images similar to the training data.

The project specifically evaluates **generalization to generators outside the original CIFAKE distribution**.

### Training

```text
CIFAKE training set
        +
1,000 Defactify training samples
        ↓
Mixed training set
        ↓
EfficientNet-B0
```

The mixed training set contained:

- 90,000 CIFAKE training images
- 1,000 Defactify training images
- **91,000 training images total**

CIFAKE validation data was kept separate for model selection.

### Held-out evaluation

The final unseen-generator evaluation used **600 Defactify images**:

- 100 real
- 100 Stable Diffusion 2.1
- 100 SDXL
- 100 Stable Diffusion 3
- 100 DALL-E 3
- 100 Midjourney 6

This evaluation set was not used for threshold tuning.

---

# 📈 Results

## Held-out Defactify evaluation

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8790** |
| Macro-F1 | **0.7303** |
| Accuracy | **82.00%** |
| False Positive Rate | **27.00%** |
| False Negative Rate | **16.20%** |

### Confusion matrix

```text
                  Predicted
                Real      AI
Actual Real      73       27
Actual AI        81      419
```

---

## Generator-wise detection

| Generator | Detection Accuracy |
|---|---:|
| DALL-E 3 | **96.0%** |
| Midjourney 6 | **85.0%** |
| SDXL | **83.0%** |
| Stable Diffusion 2.1 | **82.0%** |
| Stable Diffusion 3 | **73.0%** |

The variation across generators is important: it demonstrates that AI-image detection is a **generalization problem**, not simply an in-distribution classification problem.

---

## Baseline comparison

The original CIFAKE-only baseline performed strongly on its familiar distribution but generalized poorly to the unseen Defactify distribution.

On the same 600-image held-out evaluation:

| Model | ROC-AUC | Macro-F1 | Accuracy | FPR |
|---|---:|---:|---:|---:|
| CIFAKE-only baseline | 0.6174 | 0.4545 | 83.33% | 100.00% |
| Mixed-data model | **0.8790** | **0.7303** | 82.00% | **27.00%** |

The mixed-data model improved held-out ROC-AUC by **+0.2616 absolute** and substantially reduced false positives.

Accuracy alone is not sufficient here because the baseline predicted every image as AI on this evaluation set.

---

# 🛡️ Robustness Evaluation

SignalScope was evaluated under several common image degradations using the held-out Defactify evaluation set.

| Condition | Accuracy | Macro-F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|
| Original | 82.00% | 0.7303 | 0.8788 | 27.00% | 16.20% |
| JPEG quality 50 | 75.00% | 0.6710 | 0.8591 | 22.00% | 25.60% |
| Resize to 112 | 84.33% | 0.6052 | 0.7720 | 80.00% | 2.80% |
| Gaussian blur | 82.83% | 0.6738 | 0.7973 | 58.00% | 9.00% |
| Brightness 115% | 84.17% | 0.7419 | 0.8820 | 34.00% | 12.20% |

These results are reported as degradation-vs-performance evidence rather than as a claim of perfect robustness.

Detailed results are available in:

```text
reports/robustness/robustness_results.csv
```

---

# 🔎 Explainability

SignalScope uses **Grad-CAM** to generate a visual heatmap for the detector's prediction.

The purpose is to show which spatial regions contributed most strongly to the model's decision.

The heatmap should be treated as a model-attribution signal, not as a guaranteed human-readable proof of the exact reason an image was generated.

---

# 🧬 Generator Attribution

The attribution model predicts the likely source class among:

```text
Real
Stable Diffusion 2.1
SDXL
Stable Diffusion 3
DALL-E 3
Midjourney 6
```

This module is intended as an additional signal after the binary real-vs-AI assessment.

It should not be interpreted as definitive proof of which generator produced an image.

---

# 🧾 Provenance

SignalScope checks for two types of provenance information:

### EXIF

The backend can inspect selected metadata such as:

- Camera make
- Camera model
- Software
- Date/time
- Orientation
- Whether GPS metadata is present

GPS coordinates themselves are **not exposed by the application**.

### C2PA / Content Credentials

SignalScope attempts to read C2PA Content Credentials from the original image bytes before image preprocessing.

Possible outcomes include:

- C2PA credentials detected
- C2PA not detected
- C2PA could not be read

> Absence of C2PA or EXIF metadata does **not** mean that an image is AI-generated.

---

# 🖥️ Project Structure

```text
SignalScope/
│
├── app/
│   └── backend.py                  # FastAPI inference API
│
├── src/
│   ├── models/                    # Main detector
│   ├── attribution/               # Generator attribution
│   ├── explainability/            # Grad-CAM
│   ├── provenance/                # EXIF + C2PA
│   ├── evaluation/                # Evaluation scripts
│   ├── robustness/                # Robustness evaluation
│   └── data/                      # Dataset/data utilities
│
├── model/
│   ├── best_efficientnet_b0_mixed.pth
│   └── generator_attribution_30k.pth
│
├── frontend/
│   ├── client/
│   ├── package.json
│   └── ...
│
├── reports/
│   ├── model_report.md
│   └── robustness/
│       └── robustness_results.csv
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-deploy.txt
├── config.yaml
└── README.md
```

---

# 🚀 Quick Start — Run the Full Application Locally

The trained model checkpoints are already included in the repository, so **you do not need to retrain the model or download the training datasets just to run inference**.

## 1. Clone the repository

```bash
git clone https://github.com/adit301206/SignalScope.git
cd SignalScope
```

Checkout the final project branch if required:

```bash
git checkout main
```

---

## 2. Create a Python environment

Python **3.12** is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install backend dependencies

For the complete inference backend:

```bash
pip install -r requirements-deploy.txt
```

If PyTorch needs to be installed separately on your machine, install a compatible CPU or CUDA build from the official PyTorch instructions before running the backend.

---

## 4. Start the backend

From the repository root:

```bash
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

The backend automatically uses:

```text
CUDA → if a compatible GPU is available
CPU  → otherwise
```

---

# 🌐 Start the Frontend

Open a second terminal.

```bash
cd frontend
npm install
```

Create or verify:

```text
frontend/.env
```

with:

```env
VITE_API_URL=http://127.0.0.1:8000
VITE_USE_MOCK=false
```

Then start the frontend:

```bash
npm run dev
```

Vite will display the local frontend URL, normally:

```text
http://localhost:3000
```

Open that address in a browser.

---

# 🧪 Judge Demo Flow

A judge can verify the main system with the following sequence:

### 1. Open SignalScope

Go to the frontend URL.

### 2. Upload an image

Supported formats:

```text
JPG
PNG
WEBP
BMP
```

### 3. Run analysis

SignalScope returns:

- Real / AI-generated likelihood
- Confidence
- Class probabilities
- Grad-CAM heatmap
- Generator attribution
- EXIF information
- C2PA information

### 4. Inspect model evidence

The **Insights** page contains the held-out evaluation results and unseen-generator performance.

The **Robustness** page contains degradation results.

The **How It Works** page explains the analysis pipeline.

---

# 🔌 API

## Health

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

The device value will be `cpu` when CUDA is unavailable.

## Prediction

```http
POST /predict
```

Send the image as multipart form data using the `file` field.

Example with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

The response includes the model verdict, confidence/probabilities, Grad-CAM output, generator attribution, and provenance information.

---

# 🧪 Reproducing Evaluation

The repository also contains scripts for training and evaluation.

The main training/evaluation utilities are located under:

```text
src/
```

and the training entry points are in the repository root / evaluation modules.

For judging the submitted system, **use the included trained checkpoints** rather than retraining from scratch.

The reported results were produced from the documented held-out evaluation procedure and should not be reproduced by tuning the decision threshold on the 600-image test set.

---

# 📄 Reports

The repository contains the model report:

```text
reports/model_report.md
```

and robustness results:

```text
reports/robustness/robustness_results.csv
```

These provide the detailed experiment and evaluation record behind the results shown in the application.

---

# ⚠️ Known Limitations

SignalScope is intentionally presented as a probabilistic detection system.

Important limitations include:

1. **Unseen-generator performance varies.**  
   The model performs differently across DALL-E 3, Midjourney 6, SDXL, SD2.1 and SD3.

2. **Image degradation can affect performance.**  
   Compression, resizing and blur can change the detector's behavior.

3. **Grad-CAM is not proof.**  
   A highlighted region indicates model attribution, not a guaranteed causal explanation.

4. **Generator attribution is probabilistic.**  
   It should not be treated as forensic proof of the exact generation system.

5. **Metadata is incomplete by nature.**  
   Missing EXIF or C2PA information does not establish that an image is synthetic.

6. **The system should not be used as the sole basis for consequential authenticity decisions.**

---

# 🔐 Responsible AI

SignalScope deliberately uses language such as:

> **Likely AI-generated**

rather than:

> **This image is definitely AI-generated.**

The goal is to communicate model uncertainty and reduce overclaiming.

Provenance information is shown as supporting evidence rather than as a replacement for visual analysis.

---

# 🧰 Technology Stack

### Machine Learning

- Python
- PyTorch
- Torchvision
- EfficientNet-B0
- Scikit-learn
- OpenCV
- Grad-CAM

### Backend

- FastAPI
- Uvicorn
- Pillow
- C2PA Python
- EXIF metadata extraction

### Frontend

- React
- TypeScript
- Vite
- CSS
- Lucide icons

---

# 📚 References & Attribution

### Datasets

**CIFAKE — Real and AI-Generated Synthetic Images**

Jordan J. Bird and Ahmed Lotfi, *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, IEEE Access, 2024.

Dataset:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Repository:
https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images

**Defactify Image Dataset**

Rajarshi Roy et al., *A Comprehensive Dataset for Human vs. AI Generated Image Detection*, 2026.

Dataset:
https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset

Paper:
https://arxiv.org/abs/2601.00553

### Model

EfficientNet-B0 is used as the transfer-learning backbone with ImageNet-pretrained weights provided through Torchvision.

### Libraries

This project uses open-source libraries including PyTorch, Torchvision, FastAPI, Pillow, OpenCV, Scikit-learn, C2PA tooling, React, TypeScript, and Vite.

---

# 🎥 Demo

**Demo video:** _Add final 3–5 minute demo link here._

The demo should show:

1. A new image being uploaded
2. Real/AI verdict and confidence
3. Grad-CAM explanation
4. Generator attribution
5. Provenance information
6. Robustness/evaluation evidence
7. Responsible-AI limitations

---

# 🚀 Deployment

The project includes deployment preparation files:

```text
Dockerfile
.dockerignore
requirements-deploy.txt
```

The current submission is primarily intended to be run locally for reliable judging and reproducibility.

No deployed application is required to run the included inference system.

---

# 👥 Team

**Project:** SignalScope  
**Hackathon:** SIH 2026 — Internal Hackathon  
**Theme:** Telling Real From Synthetic in the Age of Generative Media

---

## ⭐ Final Note

SignalScope is designed around one central principle:

> **Detect, explain, verify — without overclaiming.**

A strong AI detector should not only perform well on familiar data. It should be tested against unseen generators, inspected for robustness, provide interpretable evidence, expose provenance signals when available, and communicate uncertainty responsibly.
