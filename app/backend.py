import base64
import io

from src.explainability.heatmap_utils import create_overlay

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models.model import create_model
from src.data.transforms import get_eval_transforms
from src.explainability.gradcam import GradCAM

from src.attribution.model import (
    create_attribution_model,
    GENERATOR_CLASSES,
)


# ==============================================================
# Configuration
# ==============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "model/best_efficientnet_b0_mixed.pth"
)

ATTRIBUTION_CHECKPOINT_PATH = Path(
    "model/generator_attribution_30k.pth"
)

IMAGE_SIZE = 224


# ==============================================================
# FastAPI application
# ==============================================================

app = FastAPI(
    title="SignalScope API",
    description="AI-generated image detection backend",
    version="1.0.0",
)


# ==============================================================
# CORS
# Allows the React frontend to communicate with this API.
# ==============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================
# Load model
# ==============================================================

print("=" * 60)
print("SignalScope API")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(
    f"Loading checkpoint: {CHECKPOINT_PATH}"
)

model = create_model(
    num_classes=2,
    pretrained=False,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)

if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

transform = get_eval_transforms()

# ==============================================================
# Generator Attribution Model
# ==============================================================

print(
    f"Loading attribution checkpoint: "
    f"{ATTRIBUTION_CHECKPOINT_PATH}"
)

attribution_model = create_attribution_model(
    pretrained=False
)

attribution_checkpoint = torch.load(
    ATTRIBUTION_CHECKPOINT_PATH,
    map_location=DEVICE,
)

if "model_state_dict" in attribution_checkpoint:
    attribution_state_dict = attribution_checkpoint[
        "model_state_dict"
    ]
else:
    attribution_state_dict = attribution_checkpoint

attribution_model.load_state_dict(
    attribution_state_dict
)

attribution_model = attribution_model.to(DEVICE)
attribution_model.eval()

print("Generator attribution model loaded.")

# ==============================================================
# Grad-CAM setup
# ==============================================================

target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)

print("Grad-CAM initialized.")

print("Model loaded successfully.")
print("=" * 60)


# ==============================================================
# Health check
# ==============================================================

@app.get("/")
def root():
    return {
        "name": "SignalScope API",
        "status": "running",
        "model": "EfficientNet-B0 Mixed",
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE),
    }


# ==============================================================
# Prediction endpoint
# ==============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    # ----------------------------------------------------------
    # Validate file type
    # ----------------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPG, PNG, WEBP, or BMP."
            ),
        )

    # ----------------------------------------------------------
    # Read image
    # ----------------------------------------------------------

    try:

        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    # ----------------------------------------------------------
    # Preprocess
    # ----------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # ----------------------------------------------------------
    # Model prediction
    # ----------------------------------------------------------

    # ----------------------------------------------------------
# Model prediction
# ----------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    # ----------------------------------------------------------
    # Generator Attribution Prediction
    # ----------------------------------------------------------

    with torch.no_grad():

        attribution_outputs = attribution_model(
            image_tensor
        )

        attribution_probabilities = torch.softmax(
            attribution_outputs,
            dim=1
        )[0]

    attribution_top2 = torch.topk(
        attribution_probabilities,
        k=2
    )

    attribution_predictions = []

    for score, class_index in zip(
        attribution_top2.values,
        attribution_top2.indices,
    ):
        attribution_predictions.append({
            "generator": GENERATOR_CLASSES[
                int(class_index.item())
            ],
            "confidence": round(
                float(score.item()),
                4
            ),
        })

    attribution_class = int(
        torch.argmax(
            attribution_probabilities
        ).item()
    )

    attribution_generator = GENERATOR_CLASSES[
        attribution_class
    ]

    attribution_confidence = float(
        attribution_probabilities[
            attribution_class
        ].item()
    )

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    # ----------------------------------------------------------
    # Generate Grad-CAM
    # ----------------------------------------------------------

    with torch.enable_grad():

        cam = gradcam.generate(
            image_tensor,
            predicted_class
        )

    # ----------------------------------------------------------
    # Create visual heatmap overlay
    # ----------------------------------------------------------

    heatmap_image = create_overlay(
        image,
        cam
    )

    buffer = io.BytesIO()

    heatmap_image.save(
        buffer,
        format="PNG"
    )

    heatmap_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    real_probability = float(
        probabilities[0].item()
    )

    ai_probability = float(
        probabilities[1].item()
    )

    # ----------------------------------------------------------
    # Final prediction
    # ----------------------------------------------------------

    if ai_probability >= 0.5:

        label = "AI Generated"
        confidence = ai_probability

    else:

        label = "Likely Real"
        confidence = real_probability

    # ----------------------------------------------------------
    # Response
    # ----------------------------------------------------------

    return {
        "label": label,
        "confidence": round(
            confidence,
            4
        ),
        "probability_ai": round(
            ai_probability,
            4
        ),
        "probability_real": round(
            real_probability,
            4
        ),
        "filename": file.filename,
        "heatmap_image": heatmap_base64,
        "attribution": {
            "generator": attribution_generator,
            "confidence": round(
                attribution_confidence,
                4
            ),
            "top_2": attribution_predictions,
        },
        "message": (
            "This is a likelihood assessment based on "
            "the model's learned visual patterns."
        ),
    }