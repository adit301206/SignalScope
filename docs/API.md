# API

The FastAPI application is `app.backend:app`. It has no authentication, database, queue, or persistent server-side image storage.

## `GET /`

Returns service metadata: `name`, `status`, `model`, and `device`. The default model is the bundled binary EfficientNet-B0 checkpoint.

## `GET /health`

Returns `status`, `model_loaded`, `attribution_model_loaded`, and `device`. Attribution is false when its optional checkpoint is not available.

## `POST /predict`

Submit a multipart field named `file`. JPEG, PNG, WebP, and BMP content types are accepted. The backend rejects uploads over 15 MiB after reading the body.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@path\to\image.png"
```

```json
{
  "label": "Likely Real",
  "confidence": 0.73,
  "probability_ai": 0.27,
  "probability_real": 0.73,
  "filename": "image.png",
  "heatmap_image": "iVBORw0KGgo...",
  "attribution": null,
  "message": "This is a likelihood assessment based on the model's learned visual patterns."
}
```

`heatmap_image` is base64 PNG without a data-URL prefix. When optional attribution is active, `attribution` has `generator`, `confidence`, and `top_2`; classes are real, Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney 6.

| Status | Condition |
|---|---|
| 400 | Unsupported declared content type or unreadable image |
| 413 | Upload exceeds 15 MiB |
| 500 | Unhandled model, checkpoint, processing, or server failure |

The service has no authentication, rate limiting, streaming size limit, malware scan, or production CORS policy. Do not expose it directly to untrusted public traffic without adding those controls.
