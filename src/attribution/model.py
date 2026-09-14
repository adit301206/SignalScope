import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


GENERATOR_CLASSES = [
    "real",
    "stable_diffusion_2_1",
    "stable_diffusion_xl",
    "stable_diffusion_3",
    "dalle_3",
    "midjourney_6",
]


def create_attribution_model(pretrained=True):
    """
    EfficientNet-B0 classifier for generator attribution.

    Classes:
        0 = Real
        1 = Stable Diffusion 2.1
        2 = Stable Diffusion XL
        3 = Stable Diffusion 3
        4 = DALL-E 3
        5 = Midjourney 6
    """

    if pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
    else:
        weights = None

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        len(GENERATOR_CLASSES),
    )

    return model


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    device = get_device()

    model = create_attribution_model()

    model = model.to(device)

    print("Device:", device)
    print("Model: EfficientNet-B0 Generator Attribution")
    print("Classes:", len(GENERATOR_CLASSES))
    print("Class mapping:")

    for index, name in enumerate(GENERATOR_CLASSES):
        print(f"  {index}: {name}")

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )