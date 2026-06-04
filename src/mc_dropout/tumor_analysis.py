from __future__ import annotations
import base64
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

_BRIGHT_PERCENTILE  = 85        # top 15% intensity = tumor-range brightness (seed)
_MIN_AREA           = 30        # ignore specks
_MIN_SOLIDITY       = 0.5       # rings/edges are far from convex; tumors are ~convex
_MIN_EXTENT         = 0.35      # tumors fill much of their bbox; rings barely any
_MAX_BBOX_FRAC      = 0.6       # a near-full-frame bbox is skull/brain, not a tumor
_ROI_PAD            = 15        # ROI padding around the seed blob for refinement
_MAX_GROWTH         = 4.0       # refined-vs-seed area ratio above which growth is
                                # "excessive" — only rejected if ALSO skull-shaped
_OPEN_KERNEL        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_MC_AREA_SAMPLES    = 50_000
_CONTOUR_STROKE     = 2
_ZOOM_PADDING       = 20
_DOT_RADIUS         = 1
_MAX_SCATTER_DOTS   = 5_000


def tumor_mask(image: Image.Image, image_size: int = 150) -> np.ndarray:
    """Return a binary uint8 mask (0/255) of shape (image_size, image_size).

    GradCAM-free localiser validated in the segmentation audit:

        1. seed   — largest *shape-valid* bright blob (skull-safe shape filters)
        2. refine — local Otsu inside the seed ROI, keep only the component(s)
                    connected to the seed, light morphological close
        3. guard  — reject the refined mask (fall back to the closed seed) if it
                    grows excessively *and* takes on a skull-like shape

    The classifier is no longer used for localisation; `model`/`device` were
    dropped. See `gradcam_mask` for the backward-compatible shim.
    """
    gray = np.array(image.resize((image_size, image_size)).convert("L"))

    seed = _select_seed_blob(gray)
    if seed is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)

    return _refine_roi(gray, seed)


def _select_seed_blob(gray: np.ndarray) -> np.ndarray | None:
    """Largest connected bright blob that passes the skull-safety shape filters.

    A tumor is a compact, mostly-filled, sub-frame blob. The skull rim is a thin
    hollow ring (low solidity / low extent) that spans the whole frame, so it is
    rejected by shape regardless of how bright it is.
    """
    h, w = gray.shape
    thresh = float(np.percentile(gray, _BRIGHT_PERCENTILE))
    _, bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, _OPEN_KERNEL)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    best_label: int | None = None
    best_area = 0

    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < _MIN_AREA:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw >= w * _MAX_BBOX_FRAC and bh >= h * _MAX_BBOX_FRAC:
            continue
        if area / float(bw * bh) < _MIN_EXTENT:
            continue
        if _solidity((labels == i).astype(np.uint8)) < _MIN_SOLIDITY:
            continue
        if area > best_area:
            best_area, best_label = area, i

    if best_label is None:
        return None
    return (labels == best_label).astype(np.uint8) * 255


def _refine_roi(gray: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Recover the full lesion: local Otsu in the seed ROI, connected to the seed.

    The global brightness percentile only keeps the brightest *core* of the
    lesion; re-thresholding locally (Otsu) inside the seed's padded bounding box
    recovers the dimmer periphery. The skull cannot leak in because (a) the ROI
    excludes the rim and (b) we keep only components touching the seed.
    """
    h, w = gray.shape
    x, y, bw, bh = cv2.boundingRect(seed)
    x1 = max(0, x - _ROI_PAD)
    y1 = max(0, y - _ROI_PAD)
    x2 = min(w, x + bw + _ROI_PAD)
    y2 = min(h, y + bh + _ROI_PAD)

    otsu_t, _ = cv2.threshold(gray[y1:y2, x1:x2], 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = (gray >= otsu_t).astype(np.uint8) * 255
    seg[:y1] = 0
    seg[y2:] = 0
    seg[:, :x1] = 0
    seg[:, x2:] = 0

    # Keep only components connected to the seed (region-grow from the seed).
    n_labels, labels = cv2.connectedComponents(seg)
    keep = set(np.unique(labels[seed > 0])) - {0}
    refined = np.isin(labels, list(keep)).astype(np.uint8) * 255
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

    closed_seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    if refined.sum() == 0:
        return closed_seed

    # Excessive-growth guard: legitimate lesion recovery can grow several-fold
    # (the seed is only the bright core), so a raw area cap would discard the
    # biggest wins. Only reject when growth is large AND the result is
    # skull-shaped (frame-spanning bbox or collapsed solidity).
    seed_area = float((seed > 0).sum())
    grew = (refined > 0).sum() > _MAX_GROWTH * seed_area
    if grew and _is_skull_shaped(refined):
        return closed_seed
    return refined


def _solidity(component: np.ndarray) -> float:
    """area / convex-hull area of the largest contour (0 if none)."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return (cv2.contourArea(cnt) / hull_area) if hull_area > 0 else 0.0


def _is_skull_shaped(mask: np.ndarray) -> bool:
    """A frame-spanning bounding box or a low-solidity (ring-like) blob."""
    h, w = mask.shape
    _x, _y, bw, bh = cv2.boundingRect(mask)
    if bw >= w * _MAX_BBOX_FRAC and bh >= h * _MAX_BBOX_FRAC:
        return True
    return _solidity((mask > 0).astype(np.uint8)) < _MIN_SOLIDITY


def gradcam_mask(image: Image.Image, model=None, device=None,
                 image_size: int = 150) -> np.ndarray:
    """Deprecated shim. Localisation no longer uses the classifier; `model` and
    `device` are ignored. Kept so existing callers/tests keep working."""
    return tumor_mask(image, image_size=image_size)


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
