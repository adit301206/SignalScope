import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from src.data.loaders import create_dataloaders
from src.models.model import create_model, get_device


BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 1e-4

MODEL_DIR = Path("model")
EXPERIMENT_DIR = Path("experiments/runs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_probabilities = []

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        running_loss += loss.item() * images.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_probabilities.extend(
            probabilities.detach().cpu().numpy()
        )

        progress.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader.dataset)

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    predictions = (
        np.array(all_probabilities) >= 0.5
    ).astype(int)

    f1 = f1_score(
        all_labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    return epoch_loss, auc, f1, accuracy


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            running_loss += loss.item() * images.size(0)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    epoch_loss = running_loss / len(loader.dataset)

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    predictions = (
        np.array(all_probabilities) >= 0.5
    ).astype(int)

    f1 = f1_score(
        all_labels,
        predictions,
        average="macro"
    )

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    return epoch_loss, auc, f1, accuracy


def main():
    device = get_device()

    print("=" * 60)
    print("SignalScope Training")
    print("=" * 60)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=2,
    )

    model = create_model(
        num_classes=2,
        pretrained=True,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_auc = 0.0

    history = []

    for epoch in range(1, EPOCHS + 1):

        print()
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 40)

        train_loss, train_auc, train_f1, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_auc, val_f1, val_acc = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Train | "
            f"Loss: {train_loss:.4f} | "
            f"AUC: {train_auc:.4f} | "
            f"F1: {train_f1:.4f} | "
            f"Acc: {train_acc:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_loss:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"Acc: {val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_f1": train_f1,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "val_accuracy": val_acc,
            }
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_efficientnet_b0.pth",
            )

            print(
                f"✓ Best model saved "
                f"(validation AUC: {val_auc:.4f})"
            )

    with open(
        EXPERIMENT_DIR / "training_history.json",
        "w",
    ) as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()