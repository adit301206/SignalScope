from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.model import create_model
from src.data.transforms import get_eval_transforms


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "model/best_efficientnet_b0_mixed.pth"
)


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module,
        input,
        output
    ):

        self.activations = output.detach()

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor,
        class_index
    ):

        self.model.zero_grad()

        output = self.model(
            input_tensor
        )

        score = output[:, class_index].sum()

        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(
                input_tensor.shape[2],
                input_tensor.shape[3]
            ),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


def load_model():

    model = create_model(
        num_classes=2,
        pretrained=False
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


def generate_heatmap(
    image: Image.Image
):

    model = load_model()

    transform = get_eval_transforms()

    input_tensor = transform(
        image.convert("RGB")
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    # EfficientNet-B0 final convolutional feature layer
    target_layer = model.features[-1]

    gradcam = GradCAM(
        model,
        target_layer
    )

    # We need gradients for Grad-CAM
    with torch.enable_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1
            ).item()
        )

        cam = gradcam.generate(
            input_tensor,
            predicted_class
        )

    gradcam.close()

    return (
        cam,
        predicted_class,
        probabilities[0].detach().cpu().numpy()
    )