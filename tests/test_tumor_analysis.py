import io
import base64
import numpy as np
import torch
import cv2
from PIL import Image
from mc_dropout.model import CNNModel
from mc_dropout.tumor_analysis import gradcam_mask, mc_area_estimate, render_annotated_images


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


def _solid_circle_mask(size: int = 150, radius: int = 40) -> np.ndarray:
    """Binary mask with a filled circle — known area = π*r²."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    return mask


def test_mc_area_estimate_returns_required_keys():
    mask = _solid_circle_mask()
    result = mc_area_estimate(mask, n_samples=1_000)
    assert set(result.keys()) == {
        "area_px", "mc_samples_used", "hits", "points_xy", "hits_mask", "contour_pts"
    }


def test_mc_area_estimate_area_close_to_true_circle():
    import math
    radius = 40
    mask = _solid_circle_mask(radius=radius)
    result = mc_area_estimate(mask, n_samples=100_000, rng=np.random.default_rng(42))
    expected = math.pi * radius ** 2
    # Allow 5% relative error at 100k samples
    assert abs(result["area_px"] - expected) / expected < 0.05


def test_mc_area_estimate_empty_mask_returns_zero():
    empty = np.zeros((150, 150), dtype=np.uint8)
    result = mc_area_estimate(empty, n_samples=1_000)
    assert result["area_px"] == 0.0
    assert result["hits"] == 0


def test_mc_area_estimate_samples_used_matches_n():
    mask = _solid_circle_mask()
    result = mc_area_estimate(mask, n_samples=2_000)
    assert result["mc_samples_used"] == 2_000


def test_render_annotated_images_returns_two_valid_b64_pngs():
    size = 150
    mask = _solid_circle_mask(size=size)
    area_data = mc_area_estimate(mask, n_samples=500, rng=np.random.default_rng(42))

    img = Image.new("RGB", (size, size), color=(80, 80, 80))
    contour_b64, scatter_b64 = render_annotated_images(
        original_image=img,
        contour_pts=area_data["contour_pts"],
        bbox=(0, 0, size, size),
        points_xy=area_data["points_xy"],
        hits_mask=area_data["hits_mask"],
        rng=np.random.default_rng(7),
    )
    for b64 in (contour_b64, scatter_b64):
        assert isinstance(b64, str)
        raw = base64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "must be a valid PNG"

    # Verify contour was actually drawn — output must differ from plain gray crop
    plain = np.array(img.crop((0, 0, size, size)))
    contour_arr = np.array(Image.open(io.BytesIO(base64.b64decode(contour_b64))))
    assert not np.array_equal(plain, contour_arr[:, :, :3]), "contour image must differ from plain crop"


def test_render_annotated_images_none_contour_returns_none_pair():
    img = Image.new("RGB", (150, 150))
    result = render_annotated_images(
        original_image=img,
        contour_pts=None,
        bbox=(0, 0, 150, 150),
        points_xy=np.empty((0, 2), dtype=np.float32),
        hits_mask=np.array([], dtype=bool),
    )
    assert result == (None, None)
