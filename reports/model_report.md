# Model Report

## Binary detector

`src/models/model.py` creates torchvision EfficientNet-B0 and replaces its ImageNet classifier with a two-output `nn.Linear` layer. Class 0 is real and class 1 is AI-generated. Training requests `EfficientNet_B0_Weights.DEFAULT`; inference constructs the same model without downloading pretrained weights and loads a checkpoint.

| Property | Implemented value |
|---|---|
| Input | RGB image, resized to 224 × 224 |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Output | Two logits; softmax index 1 is AI probability |
| Decision | AI if probability is at least 0.5 |
| Bundled checkpoint | `src/models/best_efficientnet_b0.pth` |
| Device | CUDA when available, otherwise CPU |
| Calibration | Not implemented |

### Training methods in code

`src/training/train.py` runs one epoch with batch size 16, AdamW (learning rate `1e-4`, weight decay `1e-4`), cross-entropy loss, and selects a checkpoint by validation ROC-AUC. `src/training/train_mixed.py` runs five epochs with batch size 16, AdamW (`1e-4`, no explicit weight decay), cross-entropy loss, and selects a checkpoint by validation ROC-AUC. Neither script implements early stopping, a learning-rate scheduler, or complete deterministic seeding. `config.yaml` is not read by either script.

Training augmentation is resize, random horizontal flip, random rotation of up to 10 degrees, and ColorJitter with brightness/contrast/saturation 0.15. Evaluation/inference only resize and normalize.

### Data documented by code

CIFAKE is represented by a local manifest and is described in existing result reports as 100,000 training and 20,000 test images, with real images from CIFAR-10 and synthetic images attributed there to Stable Diffusion v1.4. The raw manifest is not tracked, so those counts cannot be independently verified from this clone.

The mixed training selector targets 1,000 Defactify training samples: 500 real and 100 for each of Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney 6. The mixed loader combines them with CIFAKE’s post-split training rows; its validation data remains CIFAKE only. The 600-image Defactify evaluation uses 100 items per listed class/generator. The helper code identifies the source as `Rajarshi-Roy-research/Defactify_Image_Dataset`.

## Explainability

Grad-CAM registers forward/backward hooks on `model.features[-1]`, globally averages class-score gradients over spatial dimensions, weights activations, applies ReLU, upscales to 224 × 224, and normalizes the map. `create_overlay` resizes it to the original image, applies OpenCV JET colouring, and blends 40% heatmap with 60% original image. It highlights regions contributing to the selected class logit; it does not identify a specific visual defect or prove causation.

Run it with:

```powershell
python -m src.explainability.generate_heatmaps --image path\to\image.jpg --output reports\generated\overlay.jpg
```

The CLI also accepts `--model`. Its raw heatmap is saved beside the overlay with `_heatmap` appended to the file stem.

## Experimental generator attribution

`src/attribution/model.py` creates another EfficientNet-B0, this time with six output classes: real, Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney 6. The 30k trainer uses 5,000 selected images per class, an 80/20 stratified split (seed 42), two head-only epochs at `1e-3`, then up to ten late-backbone fine-tuning epochs with cosine annealing (`1e-5` backbone / `1e-4` classifier). It saves the highest validation macro-F1 checkpoint.

The attribution checkpoint is not included, so this is experimental functionality rather than a fresh-clone feature. Attribution estimates a visual class and is not provenance verification.
