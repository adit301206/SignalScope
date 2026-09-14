# Setup

## What a fresh clone contains

The repository includes Python source, frontend source, recorded CSV/text results, two Grad-CAM examples, and `src/models/best_efficientnet_b0.pth`. It does not include raw datasets, processed manifests, mixed-model checkpoints, the attribution checkpoint, or experiment histories. Git ignores those locations.

Python 3.10+ is recommended. Node.js 20+ is recommended for the frontend; `frontend/package.json` specifies pnpm 10, although its checked-in `package-lock.json` also permits `npm install`.

## Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct packages imported by the repository. Select an appropriate PyTorch wheel for CUDA/CPU if required by your platform; the code falls back to CPU when CUDA is unavailable.

## Run the API and frontend

The API loads the bundled binary classifier by default and disables attribution when `model/generator_attribution_30k.pth` is missing:

```powershell
python -m uvicorn app.backend:app --reload --host 127.0.0.1 --port 8000
```

In another shell:

```powershell
Set-Location frontend
npm install
npm run dev
```

The frontend default API URL is `http://127.0.0.1:8000`. Set `VITE_API_URL` before `npm run dev` or `npm run build` to use another API origin.

## Optional checkpoints

| Variable | Default | Purpose |
|---|---|---|
| `SIGNALSCOPE_CLASSIFIER_CHECKPOINT` | `src/models/best_efficientnet_b0.pth` | Two-class EfficientNet-B0 API weights |
| `SIGNALSCOPE_ATTRIBUTION_CHECKPOINT` | `model/generator_attribution_30k.pth` | Enables optional six-class attribution |
| `VITE_API_URL` | `http://127.0.0.1:8000` | Frontend API base URL |
| `PORT` | `3000` | Express production static-server port |

```powershell
$env:SIGNALSCOPE_ATTRIBUTION_CHECKPOINT = "C:\models\generator_attribution_30k.pth"
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
```

Both model paths must contain weights compatible with their EfficientNet-B0 classifier heads.

## Data-dependent scripts

Training/evaluation require local resources not in the clone:

- `data/processed/cifake_manifest.csv` with `path`, `label`, and `split` columns.
- Defactify train/evaluation manifests and images under `data/raw/defactify_train/` and `data/raw/defactify_eval/`.
- Defactify parquet shards under `data/raw/defactify_full/data/` for selection and robustness.
- Attribution manifest/images under `data/raw/defactify_attribution/`.

The helpers under `src/data/` write these formats. They require network access or local parquet data and do not recreate the exact recorded run: raw snapshot version and acquisition metadata are not committed.

## Production frontend build

```powershell
Set-Location frontend
npm run check
npm run build
$env:NODE_ENV = "production"
npm run start
```

This serves static frontend files on port 3000 by default. It does not run the Python API; run or deploy that process separately.
