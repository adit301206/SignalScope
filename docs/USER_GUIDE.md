# User Guide

1. Start the Python API and frontend as described in [setup](SETUP.md), then open the frontend URL.
2. Go to **Analyze**, choose or drag one JPEG, PNG, WebP, or BMP image under 15 MB, and select **Analyze image**.
3. Read the result as a likelihood. The confidence and probability bars come from the binary classifier’s softmax probabilities; 0.5 AI probability is the current decision threshold.
4. Use the original, heatmap, and overlay views to inspect Grad-CAM attention. Highlighted areas show regions that contributed more strongly to the selected class; they do not prove an artifact or origin.
5. If an attribution checkpoint was configured, an optional generator card shows a predicted class and top-two candidates. It is an experimental visual classification, not provenance verification.
6. Browser history keeps up to 12 analysis entries and a local thumbnail in `localStorage`. Use History to re-analyse or clear them. The backend has no history storage.

The Insights and Robustness routes present fixed results from tracked reports. They do not run a new evaluation on your uploaded image. No UI workflow exists for EXIF/C2PA, metadata provenance, image-text consistency, or adversarial analysis.
