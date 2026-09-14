import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.model import create_model, get_device
from src.data.transforms import get_eval_transforms
from src.explainability.gradcam import GradCAM


MODEL_PATH = "src/models/best_efficientnet_b0.pth"

CLASS_NAMES = {
    0: "REAL",
    1: "AI_GENERATED"
}


def load_model(device, model_path):
    model = create_model(
        num_classes=2,
        pretrained=False
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # Handle both plain state_dict and checkpoint dictionaries
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def generate_heatmap(image_path, output_path, model_path=MODEL_PATH):
    device = get_device()

    print("Device:", device)

    model = load_model(device, model_path)

    # Final convolutional layer
    target_layer = model.features[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    transform = get_eval_transforms()

    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.enable_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
        cam = gradcam.generate(input_tensor, predicted_class)

    ai_probability = probabilities[1].item()

    print("\nPrediction")
    print("-" * 40)
    print("Class:", CLASS_NAMES[predicted_class])
    print("AI probability:", f"{ai_probability:.4f}")
    print("REAL probability:", f"{probabilities[0].item():.4f}")

    # Resize CAM to original image dimensions
    height, width = original.shape[:2]

    cam = cv2.resize(
        cam,
        (width, height)
    )

    cam_uint8 = np.uint8(255 * cam)

    # Apply color map
    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    # Convert RGB image to BGR for OpenCV
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    # Overlay
    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        overlay
    )

    # Also save raw heatmap
    heatmap_path = output_path.with_name(
        output_path.stem + "_heatmap.jpg"
    )

    cv2.imwrite(
        str(heatmap_path),
        heatmap
    )

    print("\nSaved:")
    print(output_path)
    print(heatmap_path)

    gradcam.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save Grad-CAM overlay"
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="EfficientNet-B0 checkpoint path",
    )

    args = parser.parse_args()

    generate_heatmap(
        args.image,
        args.output,
        args.model,
    )


if __name__ == "__main__":
    main()
