# Testing and Verification

There is no automated test framework configuration, pytest suite, frontend test suite, CI workflow, or Docker build in the repository.

The available Python checks are executable smoke/evaluation scripts:

| Command | What it does | Required local inputs |
|---|---|---|
| `python -m src.test_pipeline` | DataLoader and binary model forward-pass smoke test | CIFAKE manifest/images; may download pretrained weights if uncached |
| `python -m src.evaluation.evaluate` | Binary CIFAKE test metrics | CIFAKE manifest/images and `model/best_efficientnet_b0.pth` |
| `python -m src.evaluation.evaluate_mixed` | Mixed model Defactify evaluation | Defactify eval images/manifest and mixed checkpoint |
| `python -m src.robustness.training.test_pipeline` | Defactify transform evaluation | Local Defactify parquet shards and mixed checkpoint |
| `python -m src.attribution.evaluate_30k` | Six-class attribution evaluation | Defactify eval images/manifest and attribution checkpoint |
| `npm run check` in `frontend/` | TypeScript type check | Installed frontend dependencies |
| `npm run build` in `frontend/` | Vite and Express production build | Installed frontend dependencies |

Run from the repository root unless the command explicitly says `frontend/`. The data/model prerequisites are absent from the clone, so data-dependent scripts are expected to stop with `FileNotFoundError` until supplied. Current gaps include API route tests, upload validation tests, Grad-CAM assertions, regression tests for recorded metrics, and browser tests.
