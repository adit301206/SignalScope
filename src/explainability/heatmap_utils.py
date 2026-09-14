import cv2
import numpy as np
from PIL import Image


def create_overlay(image: Image.Image, heatmap: np.ndarray):
    """
    Create a visual Grad-CAM overlay.

    Args:
        image: Original PIL image.
        heatmap: Grad-CAM array with values between 0 and 1.

    Returns:
        PIL Image containing the heatmap overlay.
    """

    # Convert PIL image to RGB NumPy array
    original = np.array(
        image.convert("RGB")
    )

    # Resize Grad-CAM to original image dimensions
    heatmap_resized = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    # Convert 0-1 heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply color map
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR, convert to RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Blend original image + heatmap
    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return Image.fromarray(
        overlay
    )