from __future__ import annotations
import base64
import io
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
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
