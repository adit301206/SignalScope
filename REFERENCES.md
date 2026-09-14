# References

## Data and models

- **Defactify Image Dataset** — `Rajarshi-Roy-research/Defactify_Image_Dataset`, the identifier used by `src/data/download_defactify_train.py` and `src/data/download_defactify_eval.py`. Dataset terms and snapshot version are not recorded locally; verify them before redistribution or submission.
- **CIFAKE** — used by local manifests and described in `reports/baseline_results.md`. The repository does not record a canonical URL, version, licence, or acquisition date, so those details must not be inferred from this project.
- **EfficientNet** — Mingxing Tan and Quoc V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML 2019. The code uses torchvision’s `efficientnet_b0` implementation and ImageNet weights when training.
- **Grad-CAM** — Ramprasaath R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV 2017.

## Libraries

Key runtime libraries are PyTorch, torchvision, FastAPI, Uvicorn, scikit-learn, pandas, Pillow, OpenCV, PyArrow, React, Vite, and Express. Exact frontend dependencies and their versions are in `frontend/package.json` / lockfiles; Python runtime dependencies are in `requirements.txt`.

## Attribution note

SignalScope-specific orchestration, training scripts, evaluation scripts, Grad-CAM hook integration, FastAPI service, and frontend are included in this repository. The model architecture, pretrained weights, research methods, datasets, and libraries remain attributable to their respective authors and licences.
