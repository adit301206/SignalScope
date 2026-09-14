"""
SignalScope Robustness Evaluation

Evaluates the trained detector under common real-world image transformations:
- JPEG compression
- Resize
- Gaussian blur
- Brightness adjustment

Uses the held-out Defactify test split stored in Parquet files.

Defactify labels:
    Label_A:
        0 = Real
        1 = AI-generated

    Label_B:
        0 = Real
        1 = Stable Diffusion 2.1
        2 = Stable Diffusion XL
        3 = Stable Diffusion 3
        4 = DALL-E 3
        5 = Midjourney

No model retraining is performed.
"""

from pathlib import Path
import csv
import io

import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision import transforms

from src.models.model import create_model


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = Path("model/best_efficientnet_b0_mixed.pth")

# Defactify Parquet dataset
DATA_ROOT = Path("data/raw/defactify_full/data")

OUTPUT_DIR = Path("reports/robustness")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224

# We use the same balanced held-out evaluation design:
# 100 real + 100 from each of 5 AI generators = 600 images.
SAMPLES_PER_CLASS = 100

GENERATOR_NAMES = {
    0: "real",
    1: "stable_diffusion_2_1",
    2: "stable_diffusion_xl",
    3: "stable_diffusion_3",
    4: "dall_e_3",
    5: "midjourney",
}


# =========================================================
# Model
# =========================================================

def load_model():
    print("\nLoading model...")

    model = create_model(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# =========================================================
# Preprocessing
# =========================================================

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# Robustness transformations
# =========================================================

def original(image):
    return image


def jpeg_compression(image):
    """
    Re-encode image as JPEG quality 50.
    """
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=50,
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def resize_image(image):
    """
    Downscale to 112x112 and resize back to 224x224.
    """
    small = image.resize(
        (112, 112),
        Image.Resampling.LANCZOS,
    )

    return small.resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )


def blur_image(image):
    """
    Mild Gaussian blur.
    """
    return image.filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )


def brightness_image(image):
    """
    Mild brightness increase.
    """
    return ImageEnhance.Brightness(image).enhance(1.15)


TRANSFORMATIONS = {
    "original": original,
    "jpeg_quality_50": jpeg_compression,
    "resize_112": resize_image,
    "gaussian_blur": blur_image,
    "brightness_115": brightness_image,
}


# =========================================================
# Defactify image decoding
# =========================================================

def decode_image(image_value):
    """
    Decode an image stored inside a Hugging Face/Parquet
    Image feature.

    The Parquet Image column can contain:
        {"bytes": ..., "path": ...}

    It may also contain raw bytes or a path.
    """

    # Case 1: dictionary from Hugging Face Image feature
    if isinstance(image_value, dict):

        image_bytes = image_value.get("bytes")

        if image_bytes is not None:
            if isinstance(image_bytes, str):
                # Usually not expected, but handle gracefully.
                image_bytes = image_bytes.encode("latin1")

            return Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")

        image_path = image_value.get("path")

        if image_path:
            path = Path(image_path)

            if path.exists():
                return Image.open(path).convert("RGB")

    # Case 2: raw bytes
    if isinstance(image_value, (bytes, bytearray)):
        return Image.open(
            io.BytesIO(image_value)
        ).convert("RGB")

    # Case 3: a path/string
    if isinstance(image_value, str):
        path = Path(image_value)

        if path.exists():
            return Image.open(path).convert("RGB")

    raise ValueError(
        f"Unsupported image format: {type(image_value)}"
    )


# =========================================================
# Dataset discovery
# =========================================================

def find_parquet_files():
    """
    Find Defactify test Parquet shards only.
    """

    parquet_files = sorted(
        DATA_ROOT.glob("test-*.parquet")
    )

    return parquet_files


def load_evaluation_dataset():
    """
    Read the held-out Defactify test split and select:

        100 Real
        100 Stable Diffusion 2.1
        100 SDXL
        100 SD3
        100 DALL-E 3
        100 Midjourney

    Total = 600 images.

    Selection is deterministic.
    """

    parquet_files = find_parquet_files()

    if not parquet_files:
        raise RuntimeError(
            f"No Defactify test Parquet files found in: {DATA_ROOT}"
        )

    print(f"\nFound {len(parquet_files)} test Parquet files.")

    selected = {
        label_b: []
        for label_b in GENERATOR_NAMES
    }

    required_total = SAMPLES_PER_CLASS * len(GENERATOR_NAMES)

    for parquet_path in parquet_files:

        print(f"Reading: {parquet_path.name}")

        df = pd.read_parquet(parquet_path)

        required_columns = {
            "Image",
            "Label_A",
            "Label_B",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{parquet_path.name} is missing columns: {missing}"
            )

        for _, row in df.iterrows():

            label_a = int(row["Label_A"])
            label_b = int(row["Label_B"])

            if label_b not in GENERATOR_NAMES:
                continue

            # Real must have Label_A = 0.
            # AI generators must have Label_A = 1.
            if label_b == 0 and label_a != 0:
                continue

            if label_b != 0 and label_a != 1:
                continue

            if len(selected[label_b]) >= SAMPLES_PER_CLASS:
                continue

            selected[label_b].append(
                {
                    "image": row["Image"],
                    "label": 0 if label_a == 0 else 1,
                    "generator": GENERATOR_NAMES[label_b],
                }
            )

            current_total = sum(
                len(items)
                for items in selected.values()
            )

            if current_total >= required_total:
                break

        current_total = sum(
            len(items)
            for items in selected.values()
        )

        if current_total >= required_total:
            break

    # Verify balance
    print("\nSelected evaluation samples:")

    for label_b, items in selected.items():
        print(
            f"  {GENERATOR_NAMES[label_b]}: {len(items)}"
        )

    missing_classes = [
        GENERATOR_NAMES[label_b]
        for label_b, items in selected.items()
        if len(items) < SAMPLES_PER_CLASS
    ]

    if missing_classes:
        raise RuntimeError(
            "Could not collect 100 samples for: "
            + ", ".join(missing_classes)
        )

    dataset = []

    for label_b in sorted(selected):
        dataset.extend(selected[label_b])

    print(
        f"\nTotal evaluation images: {len(dataset)}"
    )

    return dataset


# =========================================================
# Prediction
# =========================================================

@torch.no_grad()
def predict(model, image):
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    logits = model(tensor)

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    ai_probability = probabilities[0, 1].item()

    prediction = int(
        ai_probability >= 0.5
    )

    return prediction, ai_probability


# =========================================================
# Evaluation
# =========================================================

def evaluate_transformation(
    model,
    dataset,
    transform_name,
    transform_fn,
):
    y_true = []
    y_pred = []
    y_prob = []

    skipped = 0

    for index, item in enumerate(dataset):

        try:
            image = decode_image(
                item["image"]
            )

            transformed = transform_fn(image)

            prediction, probability = predict(
                model,
                transformed,
            )

            y_true.append(item["label"])
            y_pred.append(prediction)
            y_prob.append(probability)

        except Exception as exc:

            skipped += 1

            print(
                f"Skipping sample {index}: {exc}"
            )

    if not y_true:
        raise RuntimeError(
            f"No valid images were evaluated for {transform_name}."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    # False positive rate:
    # real images incorrectly classified as AI.
    false_positives = sum(
        1
        for truth, prediction
        in zip(y_true, y_pred)
        if truth == 0 and prediction == 1
    )

    real_count = sum(
        1
        for truth in y_true
        if truth == 0
    )

    fpr = (
        false_positives / real_count
        if real_count > 0
        else 0.0
    )

    # False negative rate:
    # AI images incorrectly classified as Real.
    false_negatives = sum(
        1
        for truth, prediction
        in zip(y_true, y_pred)
        if truth == 1 and prediction == 0
    )

    ai_count = sum(
        1
        for truth in y_true
        if truth == 1
    )

    fnr = (
        false_negatives / ai_count
        if ai_count > 0
        else 0.0
    )

    return {
        "transformation": transform_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "samples": len(y_true),
        "skipped": skipped,
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("SignalScope Robustness Evaluation")
    print("=" * 70)

    print(f"Device:  {DEVICE}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATA_ROOT}")

    # -----------------------------------------------------
    # Load held-out evaluation data
    # -----------------------------------------------------

    dataset = load_evaluation_dataset()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = load_model()

    # -----------------------------------------------------
    # Evaluate transformations
    # -----------------------------------------------------

    results = []

    for name, transform_fn in TRANSFORMATIONS.items():

        print("\n" + "-" * 70)
        print(f"Evaluating: {name}")
        print("-" * 70)

        result = evaluate_transformation(
            model,
            dataset,
            name,
            transform_fn,
        )

        results.append(result)

        print(
            f"Accuracy: {result['accuracy']:.4f}"
        )

        print(
            f"Macro-F1: {result['macro_f1']:.4f}"
        )

        print(
            f"ROC-AUC:  {result['roc_auc']:.4f}"
        )

        print(
            f"FPR:      {result['fpr']:.4f}"
        )

        print(
            f"FNR:      {result['fnr']:.4f}"
        )

        print(
            f"Samples:  {result['samples']}"
        )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    output_csv = (
        OUTPUT_DIR
        / "robustness_results.csv"
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "transformation",
                "accuracy",
                "macro_f1",
                "roc_auc",
                "fpr",
                "fnr",
                "samples",
                "skipped",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 70)

    for result in results:

        print(
            f"{result['transformation']:20s} | "
            f"AUC {result['roc_auc']:.4f} | "
            f"F1 {result['macro_f1']:.4f} | "
            f"ACC {result['accuracy']:.4f}"
        )

    print("=" * 70)
    print(
        f"Results saved to: {output_csv}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()