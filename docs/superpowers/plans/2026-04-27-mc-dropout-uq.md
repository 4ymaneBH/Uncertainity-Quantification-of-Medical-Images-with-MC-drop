# MC Dropout Uncertainty Quantification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the brain-tumor-detection Jupyter notebook into an installable Python package with a training CLI and a FastAPI web server that serves MC Dropout predictions with an HTML upload UI and uncertainty histogram.

**Architecture:** `src/mc_dropout/` package with isolated modules (config, model, dataset, train, predict, api). The FastAPI app loads the model once at startup via `lifespan` and stores it in `app.state`; the `predict.py` module is pure PyTorch with no FastAPI imports so it is testable in isolation.

**Tech Stack:** Python 3.9+, PyTorch 2.x, torchvision, FastAPI, Uvicorn, Jinja2, Pillow, matplotlib, scikit-learn, PyYAML, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Create | Package metadata, entry points `mc-train` / `mc-serve` |
| `config.yaml` | Create | Runtime defaults for all subsystems |
| `requirements.txt` | Create | Pinned dependency list |
| `.gitignore` | Create | Ignore data/, models/*.pth, __pycache__, etc. |
| `src/mc_dropout/__init__.py` | Create | Package marker |
| `src/mc_dropout/config.py` | Create | `Config` dataclass + `load_config()` |
| `src/mc_dropout/model.py` | Create | `CNNModel` (single class, dropout always present) |
| `src/mc_dropout/dataset.py` | Create | `BrainTumorDataset` + `get_dataloaders()` |
| `src/mc_dropout/train.py` | Create | `train_model()` + `main()` CLI |
| `src/mc_dropout/predict.py` | Create | `mc_predict()` + `_generate_histogram()` |
| `src/mc_dropout/api/__init__.py` | Create | Package marker |
| `src/mc_dropout/api/main.py` | Create | FastAPI app factory + `lifespan` startup |
| `src/mc_dropout/api/routes.py` | Create | `GET /` HTML, `POST /predict` JSON |
| `src/mc_dropout/api/templates/index.html` | Create | Upload form, confidence gauge, histogram display |
| `tests/__init__.py` | Create | Test package marker |
| `tests/test_config.py` | Create | Config loading tests |
| `tests/test_model.py` | Create | Forward-pass shape tests |
| `tests/test_predict.py` | Create | `mc_predict` output contract tests |
| `tests/test_api.py` | Create | FastAPI route tests via TestClient |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/mc_dropout/__init__.py`
- Create: `src/mc_dropout/api/__init__.py`
- Create: `tests/__init__.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p src/mc_dropout/api/templates
mkdir -p tests
mkdir -p models
mkdir -p data
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "mc-dropout-uq"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "Pillow>=9.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "scikit-learn>=1.2",
    "PyYAML>=6.0",
    "jinja2>=3.1",
    "python-multipart>=0.0.6",
]

[project.scripts]
mc-train = "mc_dropout.train:main"
mc-serve = "mc_dropout.api.main:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Create `config.yaml`**

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

- [ ] **Step 4: Create `requirements.txt`**

```
torch>=2.0
torchvision>=0.15
fastapi>=0.100
uvicorn[standard]>=0.20
Pillow>=9.0
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.2
PyYAML>=6.0
jinja2>=3.1
python-multipart>=0.0.6
pytest>=7.0
httpx>=0.24
```

- [ ] **Step 5: Create `.gitignore`**

```
__pycache__/
*.pyc
*.pyo
.venv/
venv/
dist/
build/
*.egg-info/
data/
models/*.pth
models/training_curves.png
.env
kaggle.json
*.zip
brain_tumor_dataset/
```

- [ ] **Step 6: Create empty `__init__.py` files**

Create the following three files, each empty:
- `src/mc_dropout/__init__.py`
- `src/mc_dropout/api/__init__.py`
- `tests/__init__.py`

Also create `models/.gitkeep` (empty file so git tracks the `models/` directory).

- [ ] **Step 7: Install the package in editable mode**

```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e .
```

Expected: package installs, `mc-train` and `mc-serve` commands become available.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml config.yaml requirements.txt .gitignore src/ tests/ models/.gitkeep
git commit -m "chore: scaffold mc-dropout-uq package structure"
```

---

## Task 2: `config.py` — Configuration Loading

**Files:**
- Create: `src/mc_dropout/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import pytest
from mc_dropout.config import (
    Config, DatasetConfig, ModelConfig, TrainingConfig,
    InferenceConfig, ApiConfig, load_config,
)

def test_load_config_returns_defaults_when_file_missing():
    config = load_config("nonexistent_file_xyz.yaml")
    assert isinstance(config, Config)
    assert config.training.epochs == 5
    assert config.inference.num_mc_samples == 100
    assert config.dataset.image_size == 150

def test_load_config_reads_yaml(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("training:\n  epochs: 99\n")
    config = load_config(str(yaml_file))
    assert config.training.epochs == 99
    assert config.training.learning_rate == 0.001  # default preserved

def test_load_config_partial_section(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("dataset:\n  batch_size: 64\n")
    config = load_config(str(yaml_file))
    assert config.dataset.batch_size == 64
    assert config.dataset.image_size == 150  # default preserved
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError` — `mc_dropout.config` does not exist yet.

- [ ] **Step 3: Implement `src/mc_dropout/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    dir: str = "./data/Brain_Tumor_Detection"
    test_split: float = 0.2
    image_size: int = 150
    batch_size: int = 32


@dataclass
class ModelConfig:
    path: str = "./models/monte_carlo_trained_model.pth"
    dropout_rate: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 0.001


@dataclass
class InferenceConfig:
    num_mc_samples: int = 100
    threshold: float = 0.5


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def load_config(path: str = "config.yaml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return Config(
        dataset=DatasetConfig(**data.get("dataset", {})),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        inference=InferenceConfig(**data.get("inference", {})),
        api=ApiConfig(**data.get("api", {})),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/config.py tests/test_config.py
git commit -m "feat: add Config dataclass and load_config()"
```

---

## Task 3: `model.py` — CNN Architecture

**Files:**
- Create: `src/mc_dropout/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import torch
from mc_dropout.model import CNNModel


def test_forward_output_shape():
    model = CNNModel()
    x = torch.zeros(2, 3, 150, 150)
    out = model(x)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"


def test_output_in_probability_range():
    model = CNNModel()
    x = torch.rand(4, 3, 150, 150)
    out = model(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    model = CNNModel(dropout_rate=0.9)
    model.train()
    x = torch.ones(1, 3, 150, 150)
    out1 = model(x).item()
    out2 = model(x).item()
    assert out1 != out2, "Outputs should differ under active dropout"


def test_dropout_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = CNNModel(dropout_rate=0.9)
    model.eval()
    x = torch.ones(1, 3, 150, 150)
    with torch.no_grad():
        out1 = model(x).item()
        out2 = model(x).item()
    assert out1 == out2, "Outputs should be deterministic in eval mode"


def test_custom_image_size():
    model = CNNModel(image_size=224)
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model.py -v
```

Expected: `ImportError` — `mc_dropout.model` does not exist.

- [ ] **Step 3: Implement `src/mc_dropout/model.py`**

```python
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """3-block CNN with dropout for MC Dropout uncertainty estimation.

    image_size must match the size used during training when loading saved weights.
    Default image_size=150 matches the provided monte_carlo_trained_model.pth.
    """

    def __init__(self, dropout_rate: float = 0.5, image_size: int = 150) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        flat_size = self._compute_flat_size(image_size)
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _compute_flat_size(self, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
            dummy = self.pool(torch.relu(self.conv3(dummy)))
            return dummy.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/model.py tests/test_model.py
git commit -m "feat: add CNNModel with dynamic fc1 size and dropout"
```

---

## Task 4: `dataset.py` — Data Loading

**Files:**
- Create: `src/mc_dropout/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset.py`:

```python
import os
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from mc_dropout.config import Config, DatasetConfig
from mc_dropout.dataset import BrainTumorDataset, get_dataloaders


def make_fake_dataset(root, n_yes=4, n_no=4):
    for category, count in [("yes", n_yes), ("no", n_no)]:
        folder = os.path.join(root, category)
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (50, 50), color=(i * 30, i * 20, 100))
            img.save(os.path.join(folder, f"img_{i}.jpg"))


def test_dataset_len(tmp_path):
    make_fake_dataset(str(tmp_path))
    paths = [str(tmp_path / "yes" / f"img_{i}.jpg") for i in range(4)]
    labels = [1] * 4
    ds = BrainTumorDataset(paths, labels)
    assert len(ds) == 4


def test_dataset_returns_image_and_label(tmp_path):
    make_fake_dataset(str(tmp_path))
    paths = [str(tmp_path / "yes" / "img_0.jpg")]
    ds = BrainTumorDataset(paths, [1])
    img, label = ds[0]
    assert label == 1
    assert img is not None


def test_get_dataloaders_splits_correctly(tmp_path):
    make_fake_dataset(str(tmp_path), n_yes=8, n_no=8)
    config = Config(dataset=DatasetConfig(
        dir=str(tmp_path), test_split=0.5, image_size=50, batch_size=4
    ))
    train_loader, test_loader = get_dataloaders(config)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    total = len(train_loader.dataset) + len(test_loader.dataset)
    assert total == 16


def test_get_dataloaders_raises_on_missing_dir():
    config = Config(dataset=DatasetConfig(dir="/nonexistent/path"))
    with pytest.raises(FileNotFoundError, match="yes"):
        get_dataloaders(config)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset.py -v
```

Expected: `ImportError` — `mc_dropout.dataset` does not exist.

- [ ] **Step 3: Implement `src/mc_dropout/dataset.py`**

```python
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mc_dropout.config import Config

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    dataset_dir = Path(config.dataset.dir)
    categories = {"yes": 1, "no": 0}

    image_paths: List[str] = []
    labels: List[int] = []

    for category, label in categories.items():
        category_path = dataset_dir / category
        if not category_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {category_path}\n"
                f"Expected sub-folders 'yes' and 'no' inside: {dataset_dir}"
            )
        for img_name in os.listdir(category_path):
            image_paths.append(str(category_path / img_name))
            labels.append(label)

    size = config.dataset.image_size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels,
        test_size=config.dataset.test_split,
        random_state=42,
    )

    train_ds = BrainTumorDataset(X_train, y_train, transform)
    test_ds = BrainTumorDataset(X_test, y_test, transform)

    return (
        DataLoader(train_ds, batch_size=config.dataset.batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=config.dataset.batch_size, shuffle=False),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dataset.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/dataset.py tests/test_dataset.py
git commit -m "feat: add BrainTumorDataset and get_dataloaders()"
```

---

## Task 5: `predict.py` — MC Dropout Inference

**Files:**
- Create: `src/mc_dropout/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_predict.py`:

```python
import base64
import torch
from PIL import Image
from mc_dropout.model import CNNModel
from mc_dropout.predict import mc_predict


def _blank_image(size: int = 150) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 128, 128))


def test_mc_predict_returns_required_keys():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert set(result.keys()) == {"prediction", "mean_probability", "uncertainty", "histogram_b64"}


def test_mc_predict_prediction_label_is_valid():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert result["prediction"] in {"Tumor", "No Tumor"}


def test_mc_predict_probability_in_range():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert 0.0 <= result["mean_probability"] <= 1.0
    assert result["uncertainty"] >= 0.0


def test_mc_predict_histogram_is_valid_base64_png():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    raw = base64.b64decode(result["histogram_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "histogram_b64 should decode to a PNG"


def test_mc_predict_threshold_respected():
    model = CNNModel()
    result = mc_predict(
        _blank_image(), model, num_samples=5,
        threshold=0.0,  # everything above 0.0 is Tumor
        device=torch.device("cpu"),
    )
    assert result["prediction"] == "Tumor"

    result2 = mc_predict(
        _blank_image(), model, num_samples=5,
        threshold=1.1,  # nothing can exceed 1.1 → No Tumor
        device=torch.device("cpu"),
    )
    assert result2["prediction"] == "No Tumor"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_predict.py -v
```

Expected: `ImportError` — `mc_dropout.predict` does not exist.

- [ ] **Step 3: Implement `src/mc_dropout/predict.py`**

```python
from __future__ import annotations
import base64
import io
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mc_dropout.model import CNNModel

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def mc_predict(
    image: Image.Image,
    model: CNNModel,
    num_samples: int = 100,
    threshold: float = 0.5,
    image_size: int = 150,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Run Monte Carlo Dropout inference on a PIL image.

    Keeps the model in train() mode so dropout stays active, then runs
    num_samples stochastic forward passes and returns mean + std.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = _get_transform(image_size)(image).unsqueeze(0).to(device)
    model.train()  # keep dropout active

    samples: list[float] = []
    with torch.no_grad():
        for _ in range(num_samples):
            samples.append(model(tensor).item())

    arr = np.array(samples)
    mean_prob = float(arr.mean())
    std_dev = float(arr.std())

    return {
        "prediction": "Tumor" if mean_prob > threshold else "No Tumor",
        "mean_probability": mean_prob,
        "uncertainty": std_dev,
        "histogram_b64": _generate_histogram(arr, mean_prob, std_dev),
    }


def _generate_histogram(samples: np.ndarray, mean: float, std: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=20, color="#4f86c6", edgecolor="white", alpha=0.85)
    ax.axvline(mean, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Mean: {mean:.3f}")
    ax.axvline(mean - std, color="#f39c12", linestyle=":", linewidth=1.5)
    ax.axvline(mean + std, color="#f39c12", linestyle=":", linewidth=1.5,
               label=f"±σ: {std:.3f}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("MC Dropout Sample Distribution")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_predict.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/predict.py tests/test_predict.py
git commit -m "feat: add mc_predict() with MC Dropout and histogram generation"
```

---

## Task 6: `train.py` — Training Loop & CLI

**Files:**
- Create: `src/mc_dropout/train.py`

No isolated unit tests — training requires a real dataset. Correctness is verified by running against the real dataset.

- [ ] **Step 1: Implement `src/mc_dropout/train.py`**

```python
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from mc_dropout.config import load_config, Config
from mc_dropout.dataset import get_dataloaders
from mc_dropout.model import CNNModel


def train_model(config: Config | None = None) -> CNNModel:
    if config is None:
        config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available — training on CPU (this will be slow)")

    train_loader, test_loader = get_dataloaders(config)
    model = CNNModel(
        dropout_rate=config.model.dropout_rate,
        image_size=config.dataset.image_size,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(config.training.epochs):
        model.train()
        running_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs.view(-1) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100.0 * correct / total)

        model.eval()
        t_loss = t_correct = t_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                t_loss += criterion(outputs.view(-1), labels).item()
                predicted = (outputs.view(-1) > 0.5).float()
                t_correct += (predicted == labels).sum().item()
                t_total += labels.size(0)

        test_losses.append(t_loss / len(test_loader))
        test_accs.append(100.0 * t_correct / t_total)

        print(
            f"Epoch [{epoch + 1}/{config.training.epochs}] "
            f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.2f}% | "
            f"Test Loss:  {test_losses[-1]:.4f}  Acc: {test_accs[-1]:.2f}%"
        )

    model_path = Path(config.model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")

    _save_training_curves(train_losses, test_losses, train_accs, test_accs, model_path.parent)
    return model


def _save_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    output_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, test_losses, label="Test Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, test_accs, label="Test Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Training curves saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MC Dropout CNN for brain tumor detection")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--lr", type=float, help="Override training.learning_rate")
    parser.add_argument("--data-dir", help="Override dataset.dir")
    parser.add_argument("--model-path", help="Override model.path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.data_dir:
        config.dataset.dir = args.data_dir
    if args.model_path:
        config.model.path = args.model_path

    train_model(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI help**

```bash
python -m mc_dropout.train --help
```

Expected output includes: `--epochs`, `--lr`, `--data-dir`, `--model-path` listed as arguments.

- [ ] **Step 3: Commit**

```bash
git add src/mc_dropout/train.py
git commit -m "feat: add train_model() and mc-train CLI entry point"
```

---

## Task 7: `api/main.py` — FastAPI App Factory

**Files:**
- Create: `src/mc_dropout/api/main.py`

- [ ] **Step 1: Implement `src/mc_dropout/api/main.py`**

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI

from mc_dropout.config import load_config
from mc_dropout.model import CNNModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(config.model.path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Run `mc-train` first to generate the checkpoint."
        )

    model = CNNModel(
        dropout_rate=config.model.dropout_rate,
        image_size=config.dataset.image_size,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"Model loaded from {model_path} on {device}")

    app.state.model = model
    app.state.config = config
    app.state.device = device

    yield

    del app.state.model


def create_app() -> FastAPI:
    app = FastAPI(
        title="MC Dropout UQ — Brain Tumor Detection",
        description="Upload a brain MRI image to receive a tumor prediction with uncertainty score.",
        version="0.1.0",
        lifespan=lifespan,
    )
    from mc_dropout.api.routes import router
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn
    config = load_config()
    uvicorn.run(
        "mc_dropout.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module imports without error**

```bash
python -c "from mc_dropout.api.main import create_app; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/mc_dropout/api/main.py
git commit -m "feat: add FastAPI app factory with lifespan model loading"
```

---

## Task 8: `api/routes.py` — HTTP Endpoints

**Files:**
- Create: `src/mc_dropout/api/routes.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import io
import base64
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mc_dropout.api.main import create_app
from mc_dropout.config import Config


def _jpeg_bytes(color=(200, 100, 50), size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _make_client_with_mock_model():
    app = create_app()
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.item.return_value = 0.8

    app.state.model = mock_model
    app.state.config = Config()
    app.state.device = __import__("torch").device("cpu")

    return TestClient(app, raise_server_exceptions=False)


def test_index_returns_html():
    app = create_app()
    app.state.model = MagicMock()
    app.state.config = Config()
    app.state.device = __import__("torch").device("cpu")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_returns_503_when_model_not_loaded():
    app = create_app()
    # Do not set app.state.model — simulate missing model
    client = TestClient(app, raise_server_exceptions=False)
    img_bytes = _jpeg_bytes()
    response = client.post("/predict", files={"file": ("tumor.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 503


def test_predict_returns_422_for_non_image():
    app = create_app()
    app.state.model = MagicMock()
    app.state.config = Config()
    app.state.device = __import__("torch").device("cpu")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"not an image", "application/pdf")},
    )
    assert response.status_code == 422


def test_predict_response_schema():
    with patch("mc_dropout.api.routes.mc_predict") as mock_pred:
        mock_pred.return_value = {
            "prediction": "Tumor",
            "mean_probability": 0.82,
            "uncertainty": 0.04,
            "histogram_b64": base64.b64encode(b"fake_png").decode(),
        }
        app = create_app()
        app.state.model = MagicMock()
        app.state.config = Config()
        app.state.device = __import__("torch").device("cpu")
        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = _jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Tumor"
    assert "mean_probability" in data
    assert "uncertainty" in data
    assert "histogram_b64" in data
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_api.py -v
```

Expected: ImportError or AttributeError — `routes` does not exist yet.

- [ ] **Step 3: Implement `src/mc_dropout/api/routes.py`**

```python
from __future__ import annotations
import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from mc_dropout.predict import mc_predict

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}


class PredictionResponse(BaseModel):
    prediction: str
    mean_probability: float
    uncertainty: float
    histogram_b64: str


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — server is still starting up.")

    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded file as an image.")

    config = request.app.state.config
    device = request.app.state.device

    try:
        result = mc_predict(
            image=image,
            model=model,
            num_samples=config.inference.num_mc_samples,
            threshold=config.inference.threshold,
            image_size=config.dataset.image_size,
            device=device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictionResponse(**result)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_api.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/api/routes.py tests/test_api.py
git commit -m "feat: add GET / and POST /predict routes with error handling"
```

---

## Task 9: `templates/index.html` — Web UI

**Files:**
- Create: `src/mc_dropout/api/templates/index.html`

No automated tests — visual correctness is verified by running the server and opening a browser.

- [ ] **Step 1: Create `src/mc_dropout/api/templates/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brain Tumor Detection — MC Dropout UQ</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0f1923;
            color: #e8eaf0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2.5rem 1rem;
        }
        h1 { font-size: 1.75rem; font-weight: 700; color: #7eb8f7; }
        .subtitle { color: #8892a4; font-size: 0.875rem; margin-top: 0.25rem; margin-bottom: 2rem; }
        .card {
            background: #1a2535;
            border-radius: 14px;
            padding: 2rem;
            width: 100%;
            max-width: 580px;
            box-shadow: 0 6px 32px rgba(0,0,0,0.5);
        }

        /* Upload area */
        .upload-area {
            border: 2px dashed #3a4f6a;
            border-radius: 10px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: #7eb8f7;
            background: rgba(126,184,247,0.04);
        }
        .upload-area input[type="file"] { display: none; }
        .upload-icon { font-size: 2.75rem; display: block; margin-bottom: 0.6rem; }
        .upload-area strong { font-size: 1rem; }
        .upload-area p { color: #8892a4; font-size: 0.85rem; margin-top: 0.35rem; }
        #file-name { margin-top: 0.6rem; color: #7eb8f7; font-size: 0.82rem; min-height: 1.1em; }
        #preview {
            max-width: 100%;
            max-height: 220px;
            border-radius: 8px;
            margin-top: 1rem;
            display: none;
            object-fit: contain;
        }

        /* Button */
        .btn {
            display: block;
            width: 100%;
            margin-top: 1.25rem;
            padding: 0.9rem;
            background: #2d6fca;
            color: #fff;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover:not(:disabled) { background: #3a84e8; }
        .btn:disabled { background: #1e2e42; color: #4a6070; cursor: not-allowed; }

        /* Spinner */
        .spinner {
            display: none;
            margin: 1.25rem auto 0;
            width: 34px; height: 34px;
            border: 3px solid #2d3f58;
            border-top-color: #7eb8f7;
            border-radius: 50%;
            animation: spin 0.75s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Error */
        .error {
            display: none;
            margin-top: 1rem;
            padding: 0.85rem 1rem;
            background: rgba(255,107,107,0.08);
            border: 1px solid #ff6b6b;
            border-radius: 8px;
            color: #ff8e8e;
            font-size: 0.875rem;
        }

        /* Results */
        .results { display: none; margin-top: 1.75rem; padding-top: 1.5rem; border-top: 1px solid #24364e; }

        .diagnosis {
            font-size: 2.1rem;
            font-weight: 800;
            text-align: center;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            letter-spacing: 0.02em;
        }
        .diagnosis.tumor   { color: #ff6b6b; background: rgba(255,107,107,0.1); }
        .diagnosis.no-tumor { color: #4ade80; background: rgba(74,222,128,0.08); }

        .metric { margin-bottom: 1.1rem; }
        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6a7f9a;
            margin-bottom: 0.4rem;
        }
        .gauge-track {
            height: 10px;
            background: #243344;
            border-radius: 999px;
            overflow: hidden;
        }
        .gauge-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1e5fa8, #7eb8f7);
            transition: width 0.55s cubic-bezier(0.4,0,0.2,1);
        }
        .metric-value { font-size: 0.82rem; color: #9aadc4; margin-top: 0.3rem; }

        .histogram { margin-top: 1.25rem; }
        .histogram img { width: 100%; border-radius: 8px; display: block; }
    </style>
</head>
<body>
    <h1>Brain Tumor Detection</h1>
    <p class="subtitle">Monte Carlo Dropout &mdash; Uncertainty Quantification</p>

    <div class="card">
        <form id="upload-form">
            <div class="upload-area" id="drop-zone">
                <input type="file" id="file-input" accept="image/jpeg,image/png">
                <span class="upload-icon">🧠</span>
                <strong>Click to upload or drag &amp; drop</strong>
                <p>Supports JPG &amp; PNG brain MRI images</p>
                <div id="file-name"></div>
                <img id="preview" alt="Image preview">
            </div>
            <button type="submit" class="btn" id="submit-btn" disabled>Analyse Image</button>
        </form>

        <div class="spinner" id="spinner"></div>
        <div class="error"   id="error-box"></div>

        <div class="results" id="results">
            <div class="diagnosis" id="diagnosis"></div>

            <div class="metric">
                <div class="metric-label">Confidence &mdash; Mean Probability</div>
                <div class="gauge-track">
                    <div class="gauge-fill" id="gauge-fill" style="width:0%"></div>
                </div>
                <div class="metric-value" id="confidence-text"></div>
            </div>

            <div class="metric">
                <div class="metric-label">Uncertainty &mdash; Standard Deviation (σ)</div>
                <div class="metric-value" id="uncertainty-text"></div>
            </div>

            <div class="histogram">
                <div class="metric-label">MC Sample Distribution ({{ config.inference.num_mc_samples if config else 100 }} forward passes)</div>
                <img id="histogram-img" src="" alt="MC Dropout probability distribution">
            </div>
        </div>
    </div>

    <script>
        const fileInput  = document.getElementById('file-input');
        const dropZone   = document.getElementById('drop-zone');
        const fileNameEl = document.getElementById('file-name');
        const preview    = document.getElementById('preview');
        const submitBtn  = document.getElementById('submit-btn');
        const spinner    = document.getElementById('spinner');
        const errorBox   = document.getElementById('error-box');
        const resultsEl  = document.getElementById('results');

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

        function setFile(file) {
            fileNameEl.textContent = file.name;
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
            submitBtn.disabled = false;
            resultsEl.style.display = 'none';
            errorBox.style.display = 'none';
        }

        document.getElementById('upload-form').addEventListener('submit', async e => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            submitBtn.disabled = true;
            spinner.style.display = 'block';
            errorBox.style.display = 'none';
            resultsEl.style.display = 'none';

            const body = new FormData();
            body.append('file', fileInput.files[0]);

            try {
                const res  = await fetch('/predict', { method: 'POST', body });
                const data = await res.json();
                if (!res.ok) { showError(data.detail || 'Prediction failed.'); return; }
                showResults(data);
            } catch (_) {
                showError('Network error — is the server running?');
            } finally {
                spinner.style.display = 'none';
                submitBtn.disabled = false;
            }
        });

        function showError(msg) {
            errorBox.textContent = msg;
            errorBox.style.display = 'block';
        }

        function showResults(data) {
            const diag = document.getElementById('diagnosis');
            diag.textContent = data.prediction;
            diag.className   = 'diagnosis ' + (data.prediction === 'Tumor' ? 'tumor' : 'no-tumor');

            const pct = (data.mean_probability * 100).toFixed(1);
            document.getElementById('gauge-fill').style.width = pct + '%';
            document.getElementById('confidence-text').textContent =
                pct + '%  (' + data.mean_probability.toFixed(4) + ')';

            document.getElementById('uncertainty-text').textContent =
                'σ = ' + data.uncertainty.toFixed(4);

            document.getElementById('histogram-img').src =
                'data:image/png;base64,' + data.histogram_b64;

            resultsEl.style.display = 'block';
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Verify template is discovered by Jinja2**

```bash
python -c "
from pathlib import Path
t = Path('src/mc_dropout/api/templates/index.html')
assert t.exists(), f'Missing: {t}'
print('Template found OK')
"
```

Expected: `Template found OK`

- [ ] **Step 3: Commit**

```bash
git add src/mc_dropout/api/templates/index.html
git commit -m "feat: add HTML upload UI with confidence gauge and histogram display"
```

---

## Task 10: Full Test Suite Run & Manual Smoke Test

**Files:** No new files. Verify everything wires together.

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests in `test_config.py`, `test_model.py`, `test_predict.py`, `test_api.py`, `test_dataset.py` pass. Total: 17+ tests green.

- [ ] **Step 2: Copy the existing model checkpoint into `models/`**

The pre-trained weights are in the project root. Move them to the canonical location:

```bash
cp monte_carlo_trained_model.pth models/monte_carlo_trained_model.pth
```

- [ ] **Step 3: Start the server**

```bash
mc-serve
```

Or equivalently:

```bash
uvicorn mc_dropout.api.main:app --host 0.0.0.0 --port 8000
```

Expected output includes:
```
Model loaded from models/monte_carlo_trained_model.pth on cpu
INFO:     Uvicorn running on http://0.0.0.0:8000
```

- [ ] **Step 4: Smoke-test the API via curl**

Open a second terminal and run:

```bash
curl -s http://localhost:8000/docs | head -5
```

Expected: HTML response starting with `<!DOCTYPE html>` (Swagger UI).

- [ ] **Step 5: Open the web UI in a browser**

Navigate to `http://localhost:8000`. Verify:
- Upload area renders with the brain emoji
- "Analyse Image" button is disabled before selecting a file
- Selecting a JPG/PNG enables the button and shows a preview

- [ ] **Step 6: Upload a test image and verify results**

Upload any JPEG brain MRI (or any RGB image for a smoke test). Verify:
- Spinner appears during inference
- Diagnosis label appears (red "Tumor" or green "No Tumor")
- Confidence gauge fills proportionally to `mean_probability`
- Uncertainty σ value is displayed
- Histogram PNG renders below the metrics

- [ ] **Step 7: Test the training CLI (optional — requires dataset)**

If the dataset is available at `./data/Brain_Tumor_Detection/`:

```bash
mc-train --epochs 1 --data-dir ./data/Brain_Tumor_Detection
```

Expected: prints epoch metrics, saves `models/monte_carlo_trained_model.pth` and `models/training_curves.png`.

- [ ] **Step 8: Final commit**

```bash
git add models/monte_carlo_trained_model.pth
git commit -m "feat: complete mc-dropout-uq package — training CLI + FastAPI web UI"
```

---

## Quick Reference

```bash
# Install
pip install -e .

# Train (requires dataset at config.yaml → dataset.dir)
mc-train
mc-train --epochs 10 --lr 0.0005 --data-dir /path/to/Brain_Tumor_Detection

# Serve
mc-serve
# then open http://localhost:8000

# Run all tests
pytest tests/ -v
```
