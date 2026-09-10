import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(num_classes=2, pretrained=True):
    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    # Replace the original ImageNet classifier.
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes
    )

    return model


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    device = get_device()

    model = create_model()
    model = model.to(device)

    print("Device:", device)
    print("Model: EfficientNet-B0")
    print("Parameters:", sum(p.numel() for p in model.parameters()))