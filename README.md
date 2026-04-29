# Uncertainty Quantification of Medical Images with MC Dropout

Brain tumor detection using **Monte Carlo Dropout** for Bayesian uncertainty quantification. A CNN trained on brain MRI images provides not just a prediction, but also a confidence score and uncertainty estimate derived from multiple stochastic forward passes.

## Features

- **MC Dropout Inference** — 100 stochastic forward passes → mean probability + std deviation
- **FastAPI web server** — REST API + browser upload UI at `http://localhost:8000`
- **Training CLI** — retrain the model from scratch with `mc-train`
- **Uncertainty histogram** — server-generated distribution plot returned as base64 PNG
- **Config-driven** — all settings in `config.yaml`, overridable via CLI flags

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Get the dataset

Download the Brain Tumor Detection MRI dataset:
- Kaggle: `kaggle datasets download -d abhranta/brain-tumor-detection-mri`
- Google Drive: see `DataSet_Google_link.txt`

Extract to `data/Brain_Tumor_Detection/` so it contains `yes/` and `no/` subfolders.

### 3. Train (or use pre-trained weights)

```bash
# Train from scratch
mc-train

# Or with overrides
mc-train --epochs 10 --lr 0.0005 --data-dir ./data/Brain_Tumor_Detection

# Pre-trained weights are already at models/monte_carlo_trained_model.pth
```

### 4. Run the server

```bash
mc-serve
```

Open `http://localhost:8000` in your browser, upload a brain MRI image, and get a prediction with uncertainty.

## API

### `POST /predict`

Upload a JPEG or PNG image:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/mri.jpg"
```

Response:
```json
{
  "prediction": "Tumor",
  "mean_probability": 0.8732,
  "uncertainty": 0.0412,
  "histogram_b64": "<base64 PNG>"
}
```

### `GET /docs`

Swagger UI — interactive API documentation.

## Project Structure

```
mc-dropout-uq/
├── src/mc_dropout/
│   ├── config.py       # Config dataclass + load_config()
│   ├── model.py        # CNNModel with MC Dropout
│   ├── dataset.py      # BrainTumorDataset + get_dataloaders()
│   ├── train.py        # Training loop + mc-train CLI
│   ├── predict.py      # mc_predict() + uncertainty histogram
│   └── api/
│       ├── main.py     # FastAPI app factory + lifespan
│       ├── routes.py   # GET / and POST /predict
│       └── templates/
│           └── index.html
├── models/             # Saved .pth checkpoints
├── data/               # Dataset (gitignored)
├── tests/              # pytest test suite
├── config.yaml         # Runtime configuration
└── pyproject.toml
```

## Configuration

Edit `config.yaml` to change defaults:

```yaml
dataset:
  dir: ./data/Brain_Tumor_Detection
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
```

## How Monte Carlo Dropout Works

Standard dropout is disabled at inference time. MC Dropout keeps it active by calling `model.train()` before each forward pass. Running N passes through the same image with different random dropout masks produces a distribution of predictions — the **mean** is the final prediction and the **standard deviation** is the uncertainty.

High uncertainty (large sigma) means the model is unsure — a flag for clinical review.

## Tech Stack

Python · PyTorch · FastAPI · Uvicorn · Jinja2 · Pillow · matplotlib · scikit-learn · PyYAML
