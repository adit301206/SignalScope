# Originality and Third-Party Components

## Project-specific work in this repository

The repository contains project-specific Python training/evaluation integration, binary and six-class model head definitions, data-manifest helpers, Grad-CAM integration, FastAPI inference wiring, and a React frontend. This statement does not claim that underlying architectures, weights, libraries, datasets, or Grad-CAM are original inventions.

## Reused components

| Component | Use | Notes |
|---|---|---|
| torchvision EfficientNet-B0 | Binary and attribution model backbone | Pretrained ImageNet weights are requested during training |
| PyTorch / torchvision | Training and inference framework | Third-party libraries |
| Grad-CAM method | Visual explanation approach | Implemented locally using PyTorch hooks; method is not original to this project |
| Defactify Image Dataset | Selected binary/attribution data and evaluation helpers | Hugging Face identifier is hard-coded in download scripts |
| CIFAKE | Binary training/evaluation manifest source | Source URL/licence metadata is not committed |
| FastAPI, React, Vite, Express, Radix UI | Application/backend/frontend infrastructure | Dependency licences apply |

The repository does not include a third-party code attribution log beyond its dependency manifests. Before a formal submission, the team should verify dataset terms, add the exact CIFAKE source/licence, retain notices required by dependencies, and record any external notebooks or code used outside this repository.
