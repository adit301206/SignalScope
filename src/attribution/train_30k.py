from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.transforms import get_train_transforms, get_eval_transforms
from src.attribution.model import create_attribution_model, GENERATOR_CLASSES


# ==============================================================
# Configuration
# ==============================================================

MANIFEST_PATH = Path(
    "data/raw/defactify_attribution/manifest.csv"
)

CHECKPOINT_PATH = Path(
    "model/generator_attribution_30k.pth"
)

BATCH_SIZE = 16

# Stage 1: classifier head
HEAD_EPOCHS = 2
HEAD_LR = 1e-3

# Stage 2: fine-tuning
FINETUNE_EPOCHS = 10
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

NUM_WORKERS = 2
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================================================
# Reproducibility
# ==============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return accuracy, macro_f1


# ==============================================================
# Freeze / Unfreeze
# ==============================================================


def freeze_backbone(model):

    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_late_backbone(model):

    # Start with everything frozen.
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze the final EfficientNet blocks.
    # This allows generator-specific features to adapt
    # while preserving most pretrained features.
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # Always train classifier.
    for param in model.classifier.parameters():
        param.requires_grad = True


# ==============================================================
# Validation
# ==============================================================


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    labels_all = []
    predictions_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            labels_all.extend(
                labels.cpu().numpy()
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

    total_loss /= len(loader.dataset)

    accuracy, macro_f1 = calculate_metrics(
        labels_all,
        predictions_all
    )

    return total_loss, accuracy, macro_f1


# ==============================================================
# Main
# ==============================================================


def main():

    set_seed(SEED)

    print("\n" + "=" * 70)
    print("SignalScope - 30K Generator Attribution Training")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ----------------------------------------------------------
    # Load manifest
    # ----------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    # Build full image path.
    df["path"] = (
        "data/raw/defactify_attribution/images/"
        + df["filename"].astype(str)
    )

    print("\nDataset size:", len(df))

    print("\nClass distribution:")

    print(
        df["generator"]
        .value_counts()
        .sort_index()
    )

    # ----------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------

    expected_classes = 6

    assert len(df) == 30000, (
        f"Expected 30000 images, found {len(df)}"
    )

    assert df["label_b"].nunique() == expected_classes, (
        "Expected exactly 6 attribution classes."
    )

    class_counts = (
        df["label_b"]
        .value_counts()
        .sort_index()
    )

    print("\nLabel distribution:")

    for label, count in class_counts.items():

        print(
            f"  {label}: "
            f"{GENERATOR_CLASSES[int(label)]} "
            f"-> {count}"
        )

    assert all(class_counts == 5000), (
        "Dataset is not balanced at 5000 samples per class."
    )

    missing_images = (
        ~df["path"]
        .map(lambda p: Path(p).exists())
    ).sum()

    if missing_images > 0:

        raise FileNotFoundError(
            f"{missing_images} image files are missing."
        )

    print("\nAll image files found.")

    # ----------------------------------------------------------
    # Train / validation split
    # ----------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["label_b"]
    )

    print("\nSplit:")
    print("  Train:", len(train_df))
    print("  Validation:", len(val_df))

    # ----------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------

    train_dataset = AttributionDataset(
        train_df,
        transform=get_train_transforms()
    )

    val_dataset = AttributionDataset(
        val_df,
        transform=get_eval_transforms()
    )

    # ----------------------------------------------------------
    # DataLoaders
    # ----------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    print("\nCreating pretrained EfficientNet-B0...")

    model = create_attribution_model(
        pretrained=True
    )

    model = model.to(DEVICE)

    print(
        "Output classes:",
        len(GENERATOR_CLASSES)
    )

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------

    # Dataset is perfectly balanced, so ordinary
    # CrossEntropyLoss is appropriate.
    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------------------------
    # Checkpoint setup
    # ----------------------------------------------------------

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_f1 = -1.0
    best_epoch = 0

    # ==========================================================
    # STAGE 1
    # ==========================================================

    print("\n" + "=" * 70)
    print("STAGE 1 - TRAIN CLASSIFIER HEAD")
    print("=" * 70)

    freeze_backbone(model)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=HEAD_LR,
        weight_decay=1e-4
    )

    for epoch in range(1, HEAD_EPOCHS + 1):

        print(
            f"\nStage 1 Epoch "
            f"{epoch}/{HEAD_EPOCHS}"
        )

        model.train()

        train_loss = 0.0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

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
            train_predictions
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train Macro-F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_acc:.4f}"
        )

        print(
            f"Val Macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = epoch

            torch.save(
                {
                    "stage": 1,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": GENERATOR_CLASSES,
                },
                CHECKPOINT_PATH
            )

            print(
                "Saved best checkpoint."
            )

    # ==========================================================
    # STAGE 2
    # ==========================================================

    print("\n" + "=" * 70)
    print("STAGE 2 - FINE-TUNE LATE BACKBONE")
    print("=" * 70)

    # Continue from the best Stage 1 model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    unfreeze_late_backbone(model)

    # Separate learning rates:
    # small LR for pretrained backbone,
    # larger LR for new classifier.
    backbone_parameters = []
    classifier_parameters = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.startswith("classifier"):
            classifier_parameters.append(param)
        else:
            backbone_parameters.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LR
            }
        ],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=FINETUNE_EPOCHS
    )

    for epoch in range(1, FINETUNE_EPOCHS + 1):

        print(
            f"\nStage 2 Epoch "
            f"{epoch}/{FINETUNE_EPOCHS}"
        )

        model.train()

        train_loss = 0.0
        train_labels = []
        train_predictions = []

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

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

        scheduler.step()

        train_loss /= len(train_loader.dataset)

        train_acc, train_f1 = calculate_metrics(
            train_labels,
            train_predictions
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train Macro-F1: {train_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_acc:.4f}"
        )

        print(
            f"Val Macro-F1: {val_f1:.4f}"
        )

        print(
            f"Backbone LR: {current_lr:.2e}"
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_epoch = (
                HEAD_EPOCHS + epoch
            )

            torch.save(
                {
                    "stage": 2,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "class_names": GENERATOR_CLASSES,
                },
                CHECKPOINT_PATH
            )

            print(
                "Saved new best checkpoint."
            )

    # ==========================================================
    # Complete
    # ==========================================================

    print("\n" + "=" * 70)
    print("30K ATTRIBUTION TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro-F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()