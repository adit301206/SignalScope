# Grad-CAM Explainability

SignalScope implements Grad-CAM for the binary EfficientNet-B0 classifier. It hooks `model.features[-1]`, computes gradient-weighted activations for the predicted class, normalizes the map, and blends it with the original image using an OpenCV JET colormap.

Run from the repository root:

```powershell
python -m src.explainability.generate_heatmaps --image path\to\image.jpg --output reports\generated\overlay.jpg
```

The default model is `src/models/best_efficientnet_b0.pth`; use `--model` to select another compatible binary checkpoint. The command writes the overlay and a sibling raw heatmap with `_heatmap` appended to its name. Existing example images are in `src/explainability/results/`.

Grad-CAM highlights locations that contributed to a class score. It is not a detector of a named artifact, a grounded textual explanation, a provenance signal, or proof that a highlighted region is synthetic.
