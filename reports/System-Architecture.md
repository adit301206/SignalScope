# System Architecture

## Runtime inference

```mermaid
flowchart TD
    U[Browser user] --> F[React / Vite client]
    F -->|multipart POST /predict| A[FastAPI app.backend]
    A --> V{JPEG, PNG, WebP, BMP and <= 15 MiB?}
    V -->|No| E[HTTP 400 or 413]
    V -->|Yes| P[Pillow RGB decode]
    P --> T[224 x 224 resize, tensor, ImageNet normalization]
    T --> B[2-class EfficientNet-B0]
    B --> S[Softmax probabilities and 0.5 label threshold]
    B --> G[Grad-CAM: model.features[-1]]
    G --> O[OpenCV/Pillow overlay encoded as base64 PNG]
    X{Attribution checkpoint exists?} -->|Yes| M[6-class EfficientNet-B0]
    X -->|No| N[attribution: null]
    T --> X
    S --> R[JSON response]
    O --> R
    M --> R
    R --> F
```

The frontend uses `VITE_API_URL` or `http://127.0.0.1:8000`, stores up to 12 history entries in browser `localStorage`, and does not call a backend history endpoint. The separate Express server only serves built frontend assets in production. There is no database, queue, object storage, Redis instance, or external inference service in code.

## Training and evaluation flow

The binary data loaders read CSV manifests with at least `path` and `label`. `create_dataloaders` takes CIFAKE rows marked `train`, shuffles with seed 42, and makes a 90/10 train/validation split; rows marked `test` remain test data. Mixed training concatenates that CIFAKE training portion with a Defactify training manifest, but retains CIFAKE validation only.

`train.py` saves a plain state dictionary to `model/best_efficientnet_b0.pth`; `train_mixed.py` saves a checkpoint dictionary to `model/best_efficientnet_b0_mixed.pth`. Evaluation scripts load those corresponding formats. The tracked checkpoint instead lives at `src/models/best_efficientnet_b0.pth`, which is the API default.

Robustness evaluation creates a deterministic balanced Defactify test sample from local parquet shards and evaluates original, JPEG-quality-50, resize-to-112-and-back, Gaussian-blur-radius-1.5, and brightness-1.15 variants. Attribution training selects 5,000 samples per class, splits 80/20 with seed 42, and evaluates on a separate 600-image Defactify manifest.

## Security and deployment boundaries

The API checks declared MIME types, decodes with Pillow, and enforces a 15 MiB post-read limit. It is otherwise a local-development service: no authentication, rate limiting, malware scanning, request streaming limit, or production CORS configuration is implemented. No Dockerfile, cloud deployment definition, health-check orchestration, or model-hosting configuration is tracked.
