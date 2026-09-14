# Limitations

- The binary detector uses visual patterns learned from limited, incompletely documented local data. Results may not transfer to new generators, domains, editing tools, or sources.
- The mixed Defactify evaluation is held out by image, but the mixed training selector deliberately includes the same named generator families. It is not strict generator-held-out evidence for the mixed model.
- At the chosen 0.5 threshold, the recorded mixed experiment misclassified 27% of real Defactify images as AI and 16.2% of AI images as real. Resize and blur made false positives materially worse in the recorded robustness run.
- Raw softmax values are displayed as confidence; no calibration method or calibration evaluation is implemented.
- Grad-CAM shows class-specific model attention, not a verified artifact, causal explanation, or forensic chain of custody. No grounded textual explanation is generated.
- Attribution is experimental, has uneven per-class results, requires an untracked checkpoint, and cannot verify the actual generator or provenance of an image.
- No EXIF parsing, C2PA/Content Credentials validation, provenance ledger, reverse-image search, or image-text consistency model exists.
- No adversarial robustness/active defence evaluation exists. JPEG, resize, blur, and brightness are the only transformations tested in code.
- A fresh clone cannot reproduce training/evaluation because the datasets, most manifests, run histories, mixed checkpoint, and attribution checkpoint are absent.
- The API is suitable only for local development as tracked: it has no authentication, rate limiting, malware scan, durable storage policy, comprehensive upload hardening, or production CORS configuration.
- The frontend displays some static experiment results and labels the UI model as online without a live health check. The Express frontend server and FastAPI service are separate processes.

SignalScope should therefore be used as one analytical input alongside source context and, where available, independently verified provenance information.
