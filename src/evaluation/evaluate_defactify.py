from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torchvision import transforms

from src.models.model import create_model, get_device


MANIFEST_PATH = Path("data/raw/defactify_eval/manifest.csv")
MODEL_PATH = Path("model/best_efficientnet_b0.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = get_device()


def main():
    print("SignalScope — Defactify Cross-Generator Evaluation")
    print("=" * 55)

    # ---------------------------------------------------------
    # Load manifest
    # ---------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    print(f"Images found: {len(df)}")
    print("\nGenerator distribution:")
    print(df["generator"].value_counts())

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = create_model(
        num_classes=2,
        pretrained=False,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()

    print(f"\nDevice: {DEVICE}")
    print("Model: EfficientNet-B0")
    print("Checkpoint loaded successfully.")

    # ---------------------------------------------------------
    # Image preprocessing
    # ---------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ---------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------

    all_labels = []
    all_probs = []
    all_predictions = []
    all_generators = []

    print("\nRunning inference...")

    with torch.no_grad():

        for start in range(0, len(df), BATCH_SIZE):

            batch_df = df.iloc[
                start:start + BATCH_SIZE
            ]

            images = []
            labels = []
            generators = []

            for _, row in batch_df.iterrows():

                image_path = Path(row["path"])

                image = Image.open(
                    image_path
                ).convert("RGB")

                image = transform(image)

                images.append(image)
                labels.append(int(row["label"]))
                generators.append(row["generator"])

            images = torch.stack(images).to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            all_labels.extend(labels)
            all_probs.extend(
                probabilities.cpu().numpy()
            )
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_generators.extend(generators)

            print(
                f"Processed "
                f"{min(start + BATCH_SIZE, len(df))}"
                f"/{len(df)}"
            )

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    labels = np.array(all_labels)
    probabilities = np.array(all_probs)
    predictions = np.array(all_predictions)

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    cm = confusion_matrix(
        labels,
        predictions,
    )

    print("\n" + "=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)

    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # ---------------------------------------------------------
    # Per-generator evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("PER-GENERATOR RESULTS")
    print("=" * 55)

    results = []

    generators = sorted(
        set(all_generators)
    )

    for generator in generators:

        mask = np.array(
            [
                g == generator
                for g in all_generators
            ]
        )

        generator_labels = labels[mask]
        generator_probs = probabilities[mask]
        generator_predictions = predictions[mask]

        # Real images contain only class 0,
        # so ROC-AUC cannot be calculated for them.
        if len(np.unique(generator_labels)) == 2:
            generator_auc = roc_auc_score(
                generator_labels,
                generator_probs,
            )
        else:
            generator_auc = float("nan")

        generator_f1 = f1_score(
            generator_labels,
            generator_predictions,
            average="macro",
            zero_division=0,
        )

        generator_accuracy = accuracy_score(
            generator_labels,
            generator_predictions,
        )

        results.append({
            "generator": generator,
            "images": int(mask.sum()),
            "roc_auc": generator_auc,
            "macro_f1": generator_f1,
            "accuracy": generator_accuracy,
        })

        print(
            f"{generator:25s} | "
            f"F1: {generator_f1:.4f} | "
            f"Accuracy: {generator_accuracy:.4f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_path = Path(
        "reports/defactify_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nPer-generator results saved to: "
        f"{output_path}"
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()