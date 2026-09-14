import torch
from src.models.model import create_model, get_device


def main():
    device = get_device()

    print("=" * 60)
    print("Device")
    print("=" * 60)
    print(device)

    model = create_model(
        num_classes=2,
        pretrained=False
    )

    model = model.to(device)
    model.eval()

    print("\n" + "=" * 60)
    print("EfficientNet-B0 Model")
    print("=" * 60)

    print(model)

    print("\n" + "=" * 60)
    print("Feature Layers")
    print("=" * 60)

    print(model.features)


if __name__ == "__main__":
    main()