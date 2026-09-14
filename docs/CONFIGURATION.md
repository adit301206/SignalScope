# Configuration

## `config.yaml`

`config.yaml` declares the task, class mapping, image size, training defaults, model name, metric names, and 0.5 threshold. No Python module reads it at present. Executable scripts use constants in their own modules, so changing the YAML does not change a run.

| Key | Value | Runtime status |
|---|---:|---|
| `data.image_size` | 224 | Matches transforms, but is not read |
| labels | real=0, AI=1 | Matches binary scripts |
| `training.batch_size` | 16 | Matches current binary scripts |
| `training.epochs` | 30 | Does not match `train.py` (1) or `train_mixed.py` (5) |
| `training.learning_rate` | 0.0001 | Matches current binary scripts |
| `model.name` | `efficientnet_b0` | Matches implementation |
| `evaluation.threshold` | 0.5 | Matches evaluators and API |

Treat script constants as source of truth until configuration loading exists.

## Runtime settings

| Setting | Default | Effect |
|---|---|---|
| `SIGNALSCOPE_CLASSIFIER_CHECKPOINT` | `src/models/best_efficientnet_b0.pth` | Binary classifier weights |
| `SIGNALSCOPE_ATTRIBUTION_CHECKPOINT` | `model/generator_attribution_30k.pth` | Enables attribution only when file exists and loads |
| `VITE_API_URL` | `http://127.0.0.1:8000` | Frontend API origin |
| `PORT` | `3000` | Express production static-server port |
| `NODE_ENV` | unset | Enables Express production static path when `production` |

The backend selects CUDA if `torch.cuda.is_available()` is true, otherwise CPU. Device selection, inference threshold, CORS origins, and accepted content types are hard-coded. The server-side upload maximum is 15 MiB.

## CORS and paths

The backend allows only `localhost`/`127.0.0.1` on ports 3000 and 5173. There is no general production CORS setting. Python paths are relative to the repository root; launch Python commands there.
