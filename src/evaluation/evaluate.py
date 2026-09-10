from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.data.loaders import create_dataloaders
from src.models.model import create_model, get_device


MODEL_PATH = Path("model/best_efficientnet_b0.pth")


def main():
    device = get_device()

    print("=" * 60)
    print("SignalScope Test Evaluation")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, _, test_loader = create_dataloaders(
        batch_size=16,
        num_workers=2,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = create_model(
        num_classes=2,
        pretrained=False,
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_probabilities = []

    print("\nRunning inference on test set...")

    with torch.no_grad():
        for images, labels in tqdm(
            test_loader,
            desc="Testing",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )[:, 1]

            all_labels.extend(
                labels.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_prob = np.array(all_probabilities)

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClass mapping:")
    print("0 = REAL")
    print("1 = FAKE")

    print("=" * 60)


if __name__ == "__main__":
    main()