from __future__ import annotations
import base64
import io
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mc_dropout.model import CNNModel

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_GRADCAM_PERCENTILE = 75
_MC_AREA_SAMPLES    = 50_000
_CONTOUR_STROKE     = 2
_ZOOM_PADDING       = 20
_DOT_RADIUS         = 1
_MAX_SCATTER_DOTS   = 5_000


def gradcam_mask(
    image: Image.Image,
    model: CNNModel,
    device: torch.device,
    image_size: int = 150,
) -> np.ndarray:
    """Return a binary uint8 mask (0/255) of shape (image_size, image_size).

    Uses GradCAM on the last conv layer (conv3) to find the attention region,
    then refines with Otsu thresholding inside that region.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    # Storage for hooks
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    fwd_hook = model.conv3.register_forward_hook(
        lambda _m, _i, out: activations.append(out)
    )
    bwd_hook = model.conv3.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0])
    )

    try:
        model.eval()
        model.zero_grad()
        out = model(tensor)
        out[0, 0].backward()
    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    # GradCAM: weight channels by global-average-pooled gradient
    act  = activations[0].squeeze(0)          # (C, H, W)
    grad = gradients[0].squeeze(0)            # (C, H, W)
    weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    cam = torch.relu((act * weights).sum(dim=0))    # (H, W)

    cam_np = cam.detach().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Upsample to image_size
    cam_resized = cv2.resize(cam_np, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)

    # Coarse attention mask: top GRADCAM_PERCENTILE activations
    threshold_val = float(np.percentile(cam_resized, _GRADCAM_PERCENTILE))
    attention_mask = (cam_resized >= threshold_val).astype(np.uint8) * 255

    if attention_mask.sum() == 0:
        # Fallback: use full image
        attention_mask = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Grayscale of the resized original image
    gray = np.array(image.resize((image_size, image_size)).convert("L"))

    # Otsu threshold inside attention region
    masked_gray = cv2.bitwise_and(gray, gray, mask=attention_mask)
    _, binary = cv2.threshold(masked_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def mc_area_estimate(mask: np.ndarray, n_samples: int = _MC_AREA_SAMPLES, rng: np.random.Generator | None = None) -> dict:
    """Estimate the area of the largest contour in mask using Monte Carlo sampling.

    Returns a dict with keys:
        area_px       – estimated area in pixels (float)
        mc_samples_used – n_samples (int)
        hits          – number of points inside the contour (int)
        points_xy     – all sampled (x, y) pairs, shape (n_samples, 2) (ndarray)
        hits_mask     – bool array of length n_samples (ndarray)
        contour_pts   – the largest contour as (N, 1, 2) int32 ndarray, or None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        empty_pts = np.zeros((n_samples, 2), dtype=np.float32)
        return {
            "area_px": 0.0,
            "mc_samples_used": n_samples,
            "hits": 0,
            "points_xy": empty_pts,
            "hits_mask": np.zeros(n_samples, dtype=bool),
            "contour_pts": None,
        }

    contour = max(contours, key=cv2.contourArea)

    # Guard: degenerate contour with no area
    if cv2.contourArea(contour) == 0:
        return {
            "area_px": 0.0,
            "mc_samples_used": n_samples,
            "hits": 0,
            "points_xy": np.zeros((n_samples, 2), dtype=np.float32),
            "hits_mask": np.zeros(n_samples, dtype=bool),
            "contour_pts": contour,
        }

    h, w = mask.shape
    x, y, cw, ch = cv2.boundingRect(contour)
    # Apply padding, clamped to image bounds
    pad = _ZOOM_PADDING
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)
    bbox_area = float((x2 - x1) * (y2 - y1))

    # Random points inside the bounding box
    if rng is None:
        rng = np.random.default_rng()
    xs = rng.uniform(x1, x2, n_samples).astype(np.float32)
    ys = rng.uniform(y1, y2, n_samples).astype(np.float32)
    points_xy = np.stack([xs, ys], axis=1)

    # Test each point against the contour polygon
    hits_mask = np.array(
        [cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0
         for px, py in points_xy],
        dtype=bool,
    )
    hits = int(hits_mask.sum())
    area_px = (hits / n_samples) * bbox_area if n_samples > 0 else 0.0

    return {
        "area_px": area_px,
        "mc_samples_used": n_samples,
        "hits": hits,
        "points_xy": points_xy,
        "hits_mask": hits_mask,
        "contour_pts": contour,
    }
