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

_GRADCAM_PERCENTILE = 85
_BRIGHT_PERCENTILE  = 85        # top 15% intensity = tumor-range brightness
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

    # Peak of GradCAM heatmap — the exact pixel the model focused on most
    peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

    # Grayscale of the resized original image
    gray = np.array(image.resize((image_size, image_size)).convert("L"))

    # High-intensity threshold: tumors appear as the brightest white region in MRI
    bright_thresh = float(np.percentile(gray, _BRIGHT_PERCENTILE))
    _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    # Find the bright connected component nearest to the GradCAM peak
    # (the skull is large and far from the peak; the tumor is compact and on-peak)
    best_mask = _pick_component_near(bright_mask, peak_x, peak_y)

    if best_mask is not None:
        return best_mask

    # Fallback: return the full bright mask so downstream always has something
    return bright_mask


def _pick_component_near(binary: np.ndarray, peak_x: int, peak_y: int) -> np.ndarray | None:
    """Return the connected component whose centroid is closest to (peak_x, peak_y).

    Skips components that are too small (noise) or whose centroid is more than
    half the image width away from the peak (clearly not the tumor).
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    h, w = binary.shape
    min_area = 30
    max_dist = w * 0.6   # ignore blobs far from the model's focus point

    best_label: int | None = None
    best_dist = float("inf")

    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        dist = np.hypot(cx - peak_x, cy - peak_y)
        if dist < best_dist and dist < max_dist:
            best_dist = dist
            best_label = i

    if best_label is None:
        return None
    return (labels == best_label).astype(np.uint8) * 255


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


def render_annotated_images(
    original_image: Image.Image,
    contour_pts: np.ndarray | None,
    bbox: Tuple[int, int, int, int],
    points_xy: np.ndarray,
    hits_mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Tuple[str | None, str | None]:
    """Produce two base64-encoded PNG strings for the result panel.

    contour_b64: zoomed crop with cyan contour stroke.
    scatter_b64: same crop + green (hit) / red (miss) MC dots overlaid.

    Returns (None, None) if contour_pts is None.
    """
    if contour_pts is None:
        return None, None

    x1, y1, x2, y2 = bbox
    # Crop from original image (resized to match mask dimensions used in analysis)
    crop = original_image.crop((x1, y1, x2, y2))
    crop_cv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)

    # Shift contour coordinates to crop-local space
    shifted = contour_pts - np.array([[[x1, y1]]], dtype=contour_pts.dtype)

    # ── Contour image ──────────────────────────────────────────
    contour_img = crop_cv.copy()
    cv2.drawContours(contour_img, [shifted], -1, (0, 255, 255), _CONTOUR_STROKE)  # BGR cyan
    contour_b64 = _encode_bgr(contour_img)

    # ── Scatter image ──────────────────────────────────────────
    scatter_img = crop_cv.copy()
    cv2.drawContours(scatter_img, [shifted], -1, (0, 255, 255), _CONTOUR_STROKE)

    # Cap dots rendered to avoid bloating the image
    if rng is None:
        rng = np.random.default_rng()
    cap = min(len(points_xy), _MAX_SCATTER_DOTS)
    idx = rng.choice(len(points_xy), cap, replace=False) if len(points_xy) > cap else np.arange(len(points_xy))
    crop_h, crop_w = scatter_img.shape[:2]
    for i in idx:
        px = max(0, min(crop_w - 1, int(points_xy[i, 0]) - x1))
        py = max(0, min(crop_h - 1, int(points_xy[i, 1]) - y1))
        color = (0, 200, 0) if hits_mask[i] else (0, 0, 200)  # BGR
        cv2.circle(scatter_img, (px, py), _DOT_RADIUS, color, -1)

    scatter_b64 = _encode_bgr(scatter_img)

    return contour_b64, scatter_b64


def _encode_bgr(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string (converts to RGB first)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", img_rgb)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
