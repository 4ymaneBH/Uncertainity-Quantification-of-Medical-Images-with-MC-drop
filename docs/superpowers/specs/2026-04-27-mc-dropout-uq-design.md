# Design Spec: MC Dropout Uncertainty Quantification — Brain Tumor Detection

**Date:** 2026-04-27  
**Status:** Approved

---

## Overview

Refactor the existing Jupyter notebook into a production-ready Python package (`mc_dropout`) with:

- A standalone training script (`train.py`)
- A FastAPI REST server with a browser-accessible HTML upload UI
- Monte Carlo Dropout inference for uncertainty quantification
- A config-file-first configuration system with CLI overrides

---

## Repository Structure

```
mc-dropout-uq/
├── src/
│   └── mc_dropout/
│       ├── __init__.py
│       ├── config.py          # loads config.yaml → typed Config dataclass
│       ├── model.py           # CNNModel + MonteCarloCNN definitions
│       ├── dataset.py         # BrainTumorDataset, get_dataloaders()
│       ├── train.py           # train_model() + CLI entry point
│       ├── predict.py         # mc_predict() → mean, std, samples[]
│       └── api/
│           ├── __init__.py
│           ├── main.py        # FastAPI app factory + lifespan startup
│           ├── routes.py      # GET / (HTML), POST /predict (JSON + histogram)
│           └── templates/
│               └── index.html # upload form, JS gauge, histogram display
├── models/                    # .pth checkpoint files (gitignored except README)
├── data/                      # dataset directory (gitignored)
├── docs/
│   └── superpowers/specs/     # design documents
├── config.yaml                # all runtime defaults
├── pyproject.toml             # package metadata + entry points
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

| Module | Responsibility | Dependencies |
|---|---|---|
| `config.py` | Load `config.yaml`, expose `Config` dataclass; accept CLI overrides | PyYAML, dataclasses |
| `model.py` | Define `CNNModel` (training) and `MonteCarloCNN` (inference with Dropout) | torch |
| `dataset.py` | `BrainTumorDataset`, `get_dataloaders(config)` | torch, torchvision, PIL |
| `train.py` | Training loop, metrics, checkpoint save, training curve plot | model.py, dataset.py, config.py |
| `predict.py` | `mc_predict(image, model, n_samples)` → `{mean, std, samples}` | model.py, torch |
| `api/main.py` | FastAPI app factory; loads model once at startup via `lifespan` | predict.py, config.py |
| `api/routes.py` | HTTP handlers; calls `mc_predict`, generates histogram PNG, returns JSON | predict.py, matplotlib |
| `templates/index.html` | Upload form; JS fetch → renders gauge + histogram | vanilla JS |

Each module has one clear purpose. `predict.py` has zero FastAPI imports — it is fully testable in isolation.

---

## Data Flow

### Training

```
config.yaml / CLI args
      ↓
config.py → Config dataclass
      ↓
dataset.py → BrainTumorDataset → train_loader, val_loader
      ↓
model.py → CNNModel (Dropout disabled during training via model.eval() in val)
      ↓
train.py → training loop (BCELoss, Adam)
         → saves models/monte_carlo_trained_model.pth
         → saves models/training_curves.png
```

### Inference (API request)

```
POST /predict  (multipart/form-data image)
      ↓
routes.py → PIL decode
      ↓
predict.py.mc_predict():
  model.train()  # keeps Dropout active
  N forward passes → samples[]
  → mean_prob, std_dev
      ↓
routes.py:
  matplotlib histogram → in-memory PNG → base64 string
  → JSON { prediction, mean_probability, uncertainty, histogram_b64 }
      ↓
index.html (JS):
  CSS confidence gauge  (mean_probability)
  uncertainty score     (std_dev)
  <img src="data:image/png;base64,...">  (histogram)
```

---

## Configuration

### `config.yaml` (defaults)

```yaml
dataset:
  dir: ./data/Brain_Tumor_Detection
  test_split: 0.2
  image_size: 150
  batch_size: 32

model:
  path: ./models/monte_carlo_trained_model.pth
  dropout_rate: 0.5

training:
  epochs: 5
  learning_rate: 0.001

inference:
  num_mc_samples: 100
  threshold: 0.5

api:
  host: 0.0.0.0
  port: 8000
```

CLI args override any key:

```bash
python -m mc_dropout.train --epochs 20 --lr 0.0005 --data-dir /path/to/data
uvicorn mc_dropout.api.main:app --host 0.0.0.0 --port 8080
```

---

## Error Handling

| Layer | Error | Response |
|---|---|---|
| `dataset.py` | Dataset dir or `yes/`/`no/` subfolders missing | `FileNotFoundError` with descriptive message |
| `predict.py` | Model file not found | `FileNotFoundError` |
| `predict.py` | Corrupt/unreadable image | `ValueError` |
| `api/routes.py` | Non-image file uploaded | HTTP 422 |
| `api/routes.py` | Model not loaded at startup | HTTP 503 |
| `api/routes.py` | Unexpected inference error | HTTP 500 with message |
| `train.py` | CUDA unavailable | Warning logged, silently falls back to CPU |

---

## API Contract

### `GET /`
Returns `index.html` — the browser upload UI.

### `POST /predict`
- **Request:** `multipart/form-data`, field `file` = image (JPG/PNG)
- **Response:**
```json
{
  "prediction": "Tumor",
  "mean_probability": 0.8732,
  "uncertainty": 0.0412,
  "histogram_b64": "<base64-encoded PNG string>"
}
```

### `GET /docs`
Swagger UI (FastAPI auto-generated).

---

## Model Architecture

`MonteCarloCNN` — identical weights to `CNNModel`, dropout kept active at inference:

```
Input (3 × 150 × 150)
  → Conv2d(3→32, 3×3) + ReLU + MaxPool2d(2)     → Dropout(0.5)
  → Conv2d(32→64, 3×3) + ReLU + MaxPool2d(2)    → Dropout(0.5)
  → Conv2d(64→128, 3×3) + ReLU + MaxPool2d(2)   → Dropout(0.5)
  → Flatten → Linear(41472→128) + ReLU           → Dropout(0.5)
  → Linear(128→1) + Sigmoid
Output: scalar probability ∈ [0, 1]
```

Threshold: `> 0.5` → "Tumor", else "No Tumor".  
MC Dropout: `model.train()` kept active; 100 stochastic forward passes → mean + std.

---

## Entry Points (`pyproject.toml`)

```toml
[project.scripts]
mc-train = "mc_dropout.train:main"
mc-serve = "mc_dropout.api.main:main"
```

After `pip install -e .`, users run `mc-train` and `mc-serve` directly.

---

## Out of Scope

- User authentication on the API
- Multi-class classification (binary only: tumor / no tumor)
- Model versioning / experiment tracking
- Frontend framework (vanilla HTML/CSS/JS only)
- Async training or distributed data loading
