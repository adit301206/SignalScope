import torch

from src.data.loaders import create_dataloaders
from src.models.model import create_model, get_device


def main():
    device = get_device()

    print("=" * 50)
    print("SignalScope Pipeline Test")
    print("=" * 50)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
    )

    print("\nDataLoader test:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch labels: {labels}")

    images = images.to(device)
    labels = labels.to(device)

    model = create_model()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    print(f"\nModel output shape: {outputs.shape}")
    print(f"Output device: {outputs.device}")

    print("\nSUCCESS: Image → DataLoader → GPU → EfficientNet")
    print("=" * 50)


if __name__ == "__main__":
    main()