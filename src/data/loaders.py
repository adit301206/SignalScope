from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import SignalScopeDataset
from src.data.transforms import (
    get_train_transforms,
    get_eval_transforms,
)


def create_dataloaders(
    manifest_path="data/processed/cifake_manifest.csv",
    batch_size=16,
    num_workers=2,
):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run create_manifest.py first."
        )

    df = pd.read_csv(manifest_path)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Create a validation split from the training data.
    # We use a fixed seed so the split is reproducible.
    train_df = train_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    validation_fraction = 0.1
    validation_size = int(len(train_df) * validation_fraction)

    val_df = train_df.iloc[:validation_size].copy()
    train_df = train_df.iloc[validation_size:].copy()

    # Save temporary manifests.
    train_manifest = Path("data/processed/train_manifest.csv")
    val_manifest = Path("data/processed/val_manifest.csv")
    test_manifest = Path("data/processed/test_manifest.csv")

    train_df.to_csv(train_manifest, index=False)
    val_df.to_csv(val_manifest, index=False)
    test_df.to_csv(test_manifest, index=False)

    train_dataset = SignalScopeDataset(
        train_manifest,
        transform=get_train_transforms(),
    )

    val_dataset = SignalScopeDataset(
        val_manifest,
        transform=get_eval_transforms(),
    )

    test_dataset = SignalScopeDataset(
        test_manifest,
        transform=get_eval_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    return train_loader, val_loader, test_loader