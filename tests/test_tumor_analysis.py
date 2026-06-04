import numpy as np
import torch
from PIL import Image
from mc_dropout.model import CNNModel
from mc_dropout.tumor_analysis import gradcam_mask


def _blank_image(size: int = 150) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 128, 128))


def test_gradcam_mask_returns_binary_ndarray():
    model = CNNModel()
    model.eval()
    img = _blank_image()
    mask = gradcam_mask(img, model, device=torch.device("cpu"), image_size=150)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (150, 150)
    assert set(np.unique(mask)).issubset({0, 255})


def test_gradcam_mask_same_size_as_image_size():
    model = CNNModel(image_size=100)
    img = _blank_image(size=100)
    mask = gradcam_mask(img, model, device=torch.device("cpu"), image_size=100)
    assert mask.shape == (100, 100)
