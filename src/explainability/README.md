# SignalScope Explainability — Grad-CAM

This module provides visual explanations for the SignalScope
real-vs-AI-generated image classifier using Grad-CAM.

## Purpose

The explainability module generates:

- Model prediction
- REAL probability
- AI_GENERATED probability
- Grad-CAM heatmap
- Heatmap overlay on the original image

Grad-CAM helps visualize spatial regions that contribute strongly
to the model's prediction.

---

## Model

The module is designed for the SignalScope EfficientNet-B0 classifier.

Expected model configuration:

- Architecture: EfficientNet-B0
- Framework: PyTorch + torchvision
- Pretrained base: ImageNet
- Input size: 224 × 224 RGB
- Number of classes: 2
- Class 0: REAL
- Class 1: AI_GENERATED
- Decision threshold: 0.5

The Grad-CAM target is the final convolutional layer:

```text
model.features[-1][0]

This corresponds to the final convolution before global average
pooling and the classification layer.

Files
explainability/
├── __init__.py
├── gradcam.py
├── generate_heatmaps.py
├── inspect_model.py
├── results/
│   ├── real/
│   └── ai/
└── README.md
gradcam.py

Contains the Grad-CAM implementation.

It:

Registers forward and backward hooks.
Extracts feature activations.
Extracts gradients for the selected class.
Performs global average pooling on gradients.
Computes the weighted activation map.
Applies ReLU.
Normalizes the heatmap.
generate_heatmaps.py

Loads the trained EfficientNet-B0 model and generates
Grad-CAM visualizations for an input image.

inspect_model.py

Used to inspect the EfficientNet-B0 architecture and verify the
Grad-CAM target layer.

Installation

From the SignalScope project root:

C:\sih-SignalScope\SignalScope

Activate the project virtual environment:

venv\Scripts\activate

Install the required visualization packages:

python -m pip install opencv-python matplotlib pillow

PyTorch and torchvision must also be installed.

Verify PyTorch:

python -c "import torch; print(torch.__version__)"

Verify CUDA:

python -c "import torch; print(torch.cuda.is_available())"

CUDA is optional. The code can run on CPU if CUDA is unavailable.

Model Checkpoint

The trained model checkpoint is intentionally not stored in GitHub.

Place the trained EfficientNet-B0 checkpoint somewhere inside
the local project, for example:

src/models/best_efficientnet_b0_mixed.pth

The checkpoint must match the SignalScope EfficientNet-B0 architecture
and two-class output configuration.

Do NOT commit large model checkpoints to GitHub unless the team
specifically decides to use Git LFS or another model-storage solution.

Input Dataset

The dataset is also not included in this repository.

For CIFAKE testing, the expected structure is:

data/
└── raw/
    └── cifake/
        └── test/
            ├── REAL/
            └── FAKE/

The exact dataset location can differ between team members.

Only the image path supplied to the command needs to be changed.

Running Grad-CAM

Run commands from the project root:

C:\sih-SignalScope\SignalScope
REAL image
python -m src.explainability.generate_heatmaps --model "src\models\best_efficientnet_b0_mixed.pth" --image "data\raw\cifake\test\REAL\example.png" --output "src\explainability\results\real\real_01_overlay.jpg"
AI-generated image
python -m src.explainability.generate_heatmaps --model "src\models\best_efficientnet_b0_mixed.pth" --image "data\raw\cifake\test\FAKE\example.png" --output "src\explainability\results\ai\ai_01_overlay.jpg"

Replace example.png with the actual image filename.

Output

For an output path such as:

src/explainability/results/real/real_01_overlay.jpg

the script generates:

real_01_overlay.jpg
real_01_overlay_heatmap.jpg

The overlay contains the original image combined with the Grad-CAM
heatmap.

The terminal also reports:

Prediction
----------------------------------------
Class: REAL
AI probability: 0.xxxx
REAL probability: 0.xxxx
Example Workflow

For a small explainability demonstration, generate Grad-CAM results
for at least:

4 REAL images
4 AI-generated images

Example:

results/
├── real/
│   ├── real_01_overlay.jpg
│   ├── real_02_overlay.jpg
│   ├── real_03_overlay.jpg
│   └── real_04_overlay.jpg
│
└── ai/
    ├── ai_01_overlay.jpg
    ├── ai_02_overlay.jpg
    ├── ai_03_overlay.jpg
    └── ai_04_overlay.jpg
Interpretation

Grad-CAM highlights spatial regions that contribute strongly to the
selected model prediction.

The heatmap should NOT be interpreted as proof that a highlighted
region is an AI artifact.

It is an interpretability tool that helps inspect what visual
evidence the classifier is using.

Troubleshooting
Model file not found

Example:

FileNotFoundError

Check the path supplied to:

--model

Example:

dir src\models
Image file not found

Check the image path:

dir data\raw\cifake\test\REAL

or:

dir data\raw\cifake\test\FAKE
CUDA unavailable

If:

torch.cuda.is_available()

returns:

False

the model can still run using CPU.

This will generally be slower, but Grad-CAM does not require a
dedicated GPU.

Import error

Run commands from the project root:

C:\sih-SignalScope\SignalScope

Use:

python -m src.explainability.generate_heatmaps

instead of executing the Python file directly.