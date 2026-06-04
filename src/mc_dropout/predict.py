from __future__ import annotations
import base64
import io
import logging
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mc_dropout.model import CNNModel
from mc_dropout.tumor_analysis import gradcam_mask, mc_area_estimate, render_annotated_images

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_log = logging.getLogger(__name__)


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
    When prediction is Tumor, also runs tumor analysis (GradCAM + MC area).
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
    prediction = "Tumor" if mean_prob > threshold else "No Tumor"

    result: Dict[str, Any] = {
        "prediction": prediction,
        "mean_probability": mean_prob,
        "uncertainty": std_dev,
        "histogram_b64": _generate_histogram(arr, mean_prob, std_dev),
        "contour_b64": None,
        "scatter_b64": None,
        "area_px": None,
        "mc_area_samples": None,
    }

    if prediction == "Tumor":
        was_training = model.training
        try:
            resized = image.resize((image_size, image_size))
            mask = gradcam_mask(resized, model, device=device, image_size=image_size)
            area_data = mc_area_estimate(mask)
            if area_data["contour_pts"] is not None:
                h, w = mask.shape
                bbox = (0, 0, w, h)
                contour_b64, scatter_b64 = render_annotated_images(
                    original_image=resized,
                    contour_pts=area_data["contour_pts"],
                    bbox=bbox,
                    points_xy=area_data["points_xy"],
                    hits_mask=area_data["hits_mask"],
                )
                result["contour_b64"] = contour_b64
                result["scatter_b64"] = scatter_b64
            result["area_px"] = area_data["area_px"]
            result["mc_area_samples"] = area_data["mc_samples_used"]
        except Exception:
            _log.exception("Tumor analysis failed; returning classification result only")
        finally:
            if was_training:
                model.train()

    return result


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
