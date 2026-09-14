from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import SignalScopeDataset
from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)

from src.attribution.model import (
    create_attribution_model,
    GENERATOR_CLASSES,
)


# ==============================================================
# Configuration
# ==============================================================

MANIFEST_PATH = Path(
    "data/raw/defactify_train/manifest.csv"
)

CHECKPOINT_PATH = Path(
    "model/generator_attribution.pth"
)

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================================================
# Dataset
# ==============================================================

class AttributionDataset(torch.utils.data.Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["path"]

        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label_b"])

        return image, label


# ==============================================================
# Metrics
# ==============================================================

def calculate_metrics(labels, predictions):

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, macro_f1


# ==============================================================
# Main
# ==============================================================

def main():

    print("\n" + "=" * 65)
    print("SignalScope - Generator Attribution Training")
    print("=" * 65)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ----------------------------------------------------------
    # Load manifest
    # ----------------------------------------------------------

    df = pd.read_csv(MANIFEST_PATH)

    # Build full image path.
    df["path"] = (
        "data/raw/defactify_train/images/"
        + df["filename"].astype(str)
    )

    # label_b is already:
    # 0 Real
    # 1 SD2.1
    # 2 SDXL
    # 3 SD3
    # 4 DALL-E3
    # 5 Midjourney

    print("\nDataset:")
    print(df["generator"].value_counts())

    # ----------------------------------------------------------
    # Train / validation split
    # ----------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["label_b"],
    )

    print("\nTrain:", len(train_df))
    print("Validation:", len(val_df))

    # ----------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------

    train_dataset = AttributionDataset(
        train_df,
        transform=get_train_transforms(),
    )

    val_dataset = AttributionDataset(
        val_df,
        transform=get_eval_transforms(),
    )

    # ----------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    model = create_attribution_model(
        pretrained=True
    )

    model = model.to(DEVICE)

    # ----------------------------------------------------------
    # Class weights
    # ----------------------------------------------------------

    class_counts = (
        train_df["label_b"]
        .value_counts()
        .sort_index()
        .values
    )

    class_weights = (
        len(train_df)
        / (
            len(class_counts)
            * class_counts
        )
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print("\nClass weights:")

    for i, weight in enumerate(class_weights):
        print(
            f"  {GENERATOR_CLASSES[i]}: "
            f"{weight.item():.3f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    best_f1 = -1.0

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, EPOCHS + 1):

        print(
            f"\n{'=' * 20} "
            f"Epoch {epoch}/{EPOCHS} "
            f"{'=' * 20}"
        )

        # ------------------------------------------------------
        # Training
        # ------------------------------------------------------

        model.train()

        train_loss = 0.0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            train_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            train_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
            )

        train_loss /= len(train_loader.dataset)

        train_acc, train_f1 = calculate_metrics(
            train_labels,
            train_predictions,
        )

        # ------------------------------------------------------
        # Validation
        # ------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_labels = []
        val_predictions = []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True,
                )

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

                val_loss += (
                    loss.item()
                    * images.size(0)
                )

                predictions = (
                    outputs.argmax(dim=1)
                )

                val_labels.extend(
                    labels.cpu().numpy()
                )

                val_predictions.extend(
                    predictions.cpu().numpy()
                )

        val_loss /= len(val_loader.dataset)

        val_acc, val_f1 = calculate_metrics(
            val_labels,
            val_predictions,
        )

        print(
            f"\nTrain Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train Macro-F1: {train_f1:.4f}"
        )

        print(
            f"\nVal Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_acc:.4f}"
        )

        print(
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # ------------------------------------------------------
        # Save best model
        # ------------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": GENERATOR_CLASSES,
                },
                CHECKPOINT_PATH,
            )

            print(
                "\n✓ Saved best attribution model:"
            )

            print(
                CHECKPOINT_PATH
            )

    print("\n" + "=" * 65)
    print("ATTRIBUTION TRAINING COMPLETE")
    print("=" * 65)

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()  