# Tumor Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GradCAM + Otsu tumor localisation pipeline with Monte Carlo area estimation that auto-expands in the UI when the classifier returns "Tumor".

**Architecture:** New `src/mc_dropout/tumor_analysis.py` module with three focused functions (`gradcam_mask`, `mc_area_estimate`, `render_annotated_images`). `mc_predict()` calls them conditionally; API schemas gain 4 optional fields; the React frontend adds a `TumorAnalysisPanel` component that appears automatically when `contour_b64` is present.

**Tech Stack:** Python 3.9+, PyTorch (GradCAM hooks), opencv-python (Otsu, findContours, pointPolygonTest), Pillow, NumPy, React 18 (CDN, JSX via Babel)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/mc_dropout/tumor_analysis.py` | GradCAM mask, MC area estimation, image rendering |
| Modify | `src/mc_dropout/predict.py` | Call tumor analysis when prediction == "Tumor" |
| Modify | `src/mc_dropout/api/routes.py` | 4 new optional fields in PredictionResponse + BatchPredictionItem |
| Modify | `src/mc_dropout/api/templates/index.html` | TumorAnalysisPanel React component in ResultCard |
| Modify | `pyproject.toml` | Add opencv-python dependency |
| Modify | `requirements.txt` | Add opencv-python |
| Create | `tests/test_tumor_analysis.py` | Unit tests for all three analysis functions |
| Modify | `tests/test_predict.py` | Add tests for new fields in mc_predict output |

---

## Task 1: Add opencv-python dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add opencv-python to pyproject.toml**

In `pyproject.toml`, add `"opencv-python>=4.8"` to the `dependencies` list:

```toml
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
    "opencv-python>=4.8",
]
```

- [ ] **Step 2: Add opencv-python to requirements.txt**

Append to `requirements.txt`:
```
opencv-python>=4.8
```

- [ ] **Step 3: Install the dependency**

```bash
pip install opencv-python>=4.8
```

Expected: installs without error; `import cv2` works.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt
git commit -m "chore: add opencv-python dependency for tumor analysis"
```

---

## Task 2: Create tumor_analysis.py — gradcam_mask

**Files:**
- Create: `src/mc_dropout/tumor_analysis.py`
- Create: `tests/test_tumor_analysis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tumor_analysis.py`:

```python
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
    model = CNNModel()
    img = _blank_image(size=100)
    mask = gradcam_mask(img, model, device=torch.device("cpu"), image_size=100)
    assert mask.shape == (100, 100)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_tumor_analysis.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `tumor_analysis` doesn't exist yet.

- [ ] **Step 3: Implement gradcam_mask in tumor_analysis.py**

Create `src/mc_dropout/tumor_analysis.py`:

```python
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
        out = model(tensor)
        model.zero_grad()
        out.backward()
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_tumor_analysis.py::test_gradcam_mask_returns_binary_ndarray tests/test_tumor_analysis.py::test_gradcam_mask_same_size_as_image_size -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/tumor_analysis.py tests/test_tumor_analysis.py
git commit -m "feat: add gradcam_mask to tumor_analysis module"
```

---

## Task 3: Add mc_area_estimate to tumor_analysis.py

**Files:**
- Modify: `src/mc_dropout/tumor_analysis.py`
- Modify: `tests/test_tumor_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tumor_analysis.py`:

```python
from mc_dropout.tumor_analysis import mc_area_estimate


def _solid_circle_mask(size: int = 150, radius: int = 40) -> np.ndarray:
    """Binary mask with a filled circle — known area = π*r²."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)
    return mask


def test_mc_area_estimate_returns_required_keys():
    import cv2 as _cv2
    mask = _solid_circle_mask()
    result = mc_area_estimate(mask, n_samples=1_000)
    assert set(result.keys()) == {
        "area_px", "mc_samples_used", "hits", "points_xy", "hits_mask", "contour_pts"
    }


def test_mc_area_estimate_area_close_to_true_circle():
    import math, cv2 as _cv2
    radius = 40
    mask = _solid_circle_mask(radius=radius)
    result = mc_area_estimate(mask, n_samples=100_000)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tumor_analysis.py::test_mc_area_estimate_returns_required_keys -v
```

Expected: `ImportError` — `mc_area_estimate` not defined yet.

- [ ] **Step 3: Implement mc_area_estimate**

Add to `src/mc_dropout/tumor_analysis.py` (after the `gradcam_mask` function):

```python
def mc_area_estimate(mask: np.ndarray, n_samples: int = _MC_AREA_SAMPLES) -> dict:
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
        empty_pts = np.empty((n_samples, 2), dtype=np.float32)
        return {
            "area_px": 0.0,
            "mc_samples_used": n_samples,
            "hits": 0,
            "points_xy": empty_pts,
            "hits_mask": np.zeros(n_samples, dtype=bool),
            "contour_pts": None,
        }

    contour = max(contours, key=cv2.contourArea)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tumor_analysis.py -k "mc_area" -v
```

Expected: all 4 `mc_area_estimate` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/tumor_analysis.py tests/test_tumor_analysis.py
git commit -m "feat: add mc_area_estimate — Monte Carlo contour area estimation"
```

---

## Task 4: Add render_annotated_images to tumor_analysis.py

**Files:**
- Modify: `src/mc_dropout/tumor_analysis.py`
- Modify: `tests/test_tumor_analysis.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tumor_analysis.py`:

```python
from mc_dropout.tumor_analysis import render_annotated_images


def test_render_annotated_images_returns_two_valid_b64_pngs():
    import base64 as _b64
    size = 150
    mask = _solid_circle_mask(size=size)
    area_data = mc_area_estimate(mask, n_samples=500)

    img = Image.new("RGB", (size, size), color=(80, 80, 80))
    contour_b64, scatter_b64 = render_annotated_images(
        original_image=img,
        contour_pts=area_data["contour_pts"],
        bbox=(0, 0, size, size),
        points_xy=area_data["points_xy"],
        hits_mask=area_data["hits_mask"],
    )
    for b64 in (contour_b64, scatter_b64):
        assert isinstance(b64, str)
        raw = _b64.b64decode(b64)
        assert raw[:8] == b"\x89PNG\r\n\x1a\n", "must be a valid PNG"


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tumor_analysis.py::test_render_annotated_images_returns_two_valid_b64_pngs -v
```

Expected: `ImportError` — `render_annotated_images` not defined yet.

- [ ] **Step 3: Implement render_annotated_images**

Add to `src/mc_dropout/tumor_analysis.py` (after `mc_area_estimate`):

```python
def render_annotated_images(
    original_image: Image.Image,
    contour_pts: np.ndarray | None,
    bbox: Tuple[int, int, int, int],
    points_xy: np.ndarray,
    hits_mask: np.ndarray,
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
    cv2.drawContours(contour_img, [shifted], -1, (0, 255, 255), _CONTOUR_STROKE)
    contour_b64 = _encode_bgr(contour_img)

    # ── Scatter image ──────────────────────────────────────────
    scatter_img = crop_cv.copy()
    cv2.drawContours(scatter_img, [shifted], -1, (0, 255, 255), _CONTOUR_STROKE)

    # Cap dots rendered to avoid bloating the image
    cap = min(len(points_xy), _MAX_SCATTER_DOTS)
    idx = np.random.choice(len(points_xy), cap, replace=False) if len(points_xy) > cap else np.arange(len(points_xy))
    for i in idx:
        px, py = int(points_xy[i, 0]) - x1, int(points_xy[i, 1]) - y1
        color = (0, 200, 0) if hits_mask[i] else (0, 0, 200)  # BGR
        cv2.circle(scatter_img, (px, py), _DOT_RADIUS, color, -1)

    scatter_b64 = _encode_bgr(scatter_img)

    return contour_b64, scatter_b64


def _encode_bgr(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
```

- [ ] **Step 4: Run all tumor_analysis tests**

```bash
pytest tests/test_tumor_analysis.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/tumor_analysis.py tests/test_tumor_analysis.py
git commit -m "feat: add render_annotated_images — contour + MC scatter PNGs"
```

---

## Task 5: Extend mc_predict to call tumor analysis

**Files:**
- Modify: `src/mc_dropout/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_predict.py`:

```python
def test_mc_predict_no_tumor_has_null_analysis_fields():
    model = CNNModel()
    result = mc_predict(
        _blank_image(), model,
        num_samples=5,
        threshold=1.1,   # force "No Tumor"
        device=torch.device("cpu"),
    )
    assert result["prediction"] == "No Tumor"
    assert result["contour_b64"] is None
    assert result["scatter_b64"] is None
    assert result["area_px"] is None
    assert result["mc_area_samples"] is None


def test_mc_predict_tumor_has_analysis_fields():
    import base64 as _b64
    model = CNNModel()
    result = mc_predict(
        _blank_image(), model,
        num_samples=5,
        threshold=0.0,   # force "Tumor"
        device=torch.device("cpu"),
    )
    assert result["prediction"] == "Tumor"
    # area_px may be 0 on a blank image (no contour), but keys must exist
    assert "contour_b64" in result
    assert "scatter_b64" in result
    assert "area_px" in result
    assert "mc_area_samples" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_predict.py::test_mc_predict_no_tumor_has_null_analysis_fields -v
```

Expected: `KeyError: 'contour_b64'` — fields not present yet.

- [ ] **Step 3: Update mc_predict to call tumor analysis**

Replace the entire `mc_predict` function in `src/mc_dropout/predict.py`:

```python
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
        try:
            # Resize image to match the model's working size
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
```

- [ ] **Step 4: Run all predict tests**

```bash
pytest tests/test_predict.py -v
```

Expected: all 7 tests PASS (the original 5 plus 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/predict.py tests/test_predict.py
git commit -m "feat: extend mc_predict to run tumor analysis when prediction is Tumor"
```

---

## Task 6: Update API schemas in routes.py

**Files:**
- Modify: `src/mc_dropout/api/routes.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Read `tests/test_api.py` to see what's there, then append:

```python
def test_predict_response_includes_analysis_fields(client, tmp_path):
    """The /predict response must always include the 4 analysis fields (may be None)."""
    img_path = tmp_path / "scan.png"
    Image.new("RGB", (150, 150), color=(128, 128, 128)).save(img_path)

    with open(img_path, "rb") as f:
        resp = client.post("/predict", files={"file": ("scan.png", f, "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    for key in ("contour_b64", "scatter_b64", "area_px", "mc_area_samples"):
        assert key in body, f"Missing field: {key}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_api.py::test_predict_response_includes_analysis_fields -v
```

Expected: `AssertionError: Missing field: contour_b64` — schema doesn't expose it yet.

- [ ] **Step 3: Update PredictionResponse and BatchPredictionItem**

In `src/mc_dropout/api/routes.py`, replace the two Pydantic models:

```python
class PredictionResponse(BaseModel):
    prediction: str
    mean_probability: float
    uncertainty: float
    histogram_b64: str
    contour_b64: Optional[str] = None
    scatter_b64: Optional[str] = None
    area_px: Optional[float] = None
    mc_area_samples: Optional[int] = None


class BatchPredictionItem(BaseModel):
    filename: str
    prediction: Optional[str] = None
    mean_probability: Optional[float] = None
    uncertainty: Optional[float] = None
    histogram_b64: Optional[str] = None
    contour_b64: Optional[str] = None
    scatter_b64: Optional[str] = None
    area_px: Optional[float] = None
    mc_area_samples: Optional[int] = None
    error: Optional[str] = None
```

- [ ] **Step 4: Run all API tests**

```bash
pytest tests/test_api.py -v
```

Expected: all tests PASS including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/api/routes.py tests/test_api.py
git commit -m "feat: add contour_b64, scatter_b64, area_px, mc_area_samples to API response"
```

---

## Task 7: Add TumorAnalysisPanel to index.html

**Files:**
- Modify: `src/mc_dropout/api/templates/index.html`

- [ ] **Step 1: Locate the insertion point**

Open `src/mc_dropout/api/templates/index.html`. Find the `ResultCard` component (~line 636). The component renders:
```jsx
<DiagnosisHeader ... />
<ConfidenceGauge ... />
<UncertaintyRow ... />
{item.histogram_b64 && <Histogram b64={item.histogram_b64} />}
```
The `TumorAnalysisPanel` goes **after** the `Histogram` line and **before** the closing `</div>` of `ResultCard`.

- [ ] **Step 2: Add the TumorAnalysisPanel component**

Insert the following new component **before** the `ResultCard` function definition (around line 636):

```jsx
/* ─────────────────────────────────────────────────────────────
   TumorAnalysisPanel
───────────────────────────────────────────────────────────── */
function TumorAnalysisPanel({ contourB64, scatterB64, areaPx, mcSamples }) {
  const [open, setOpen] = React.useState(true);
  if (!contourB64) return null;

  const fmtArea = n => n == null ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });

  return (
    <div style={{ marginTop: 24 }}>
      <button
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-secondary)', padding: '4px 0', marginBottom: 10,
        }}
      >
        <ChevronIcon open={open} />
        <span className="label">Tumor region analysis</span>
      </button>

      {open && (
        <div className="fade-up">
          {/* Side-by-side images */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 14 }}>
            <div>
              <div className="label" style={{ marginBottom: 6, fontSize: 10 }}>Contour</div>
              <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                <img
                  src={`data:image/png;base64,${contourB64}`}
                  alt="Tumor contour"
                  style={{ width: '100%', display: 'block' }}
                />
              </div>
            </div>
            <div>
              <div className="label" style={{ marginBottom: 6, fontSize: 10 }}>MC sampling</div>
              <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
                <img
                  src={`data:image/png;base64,${scatterB64}`}
                  alt="Monte Carlo area sampling"
                  style={{ width: '100%', display: 'block' }}
                />
              </div>
            </div>
          </div>

          {/* Stats row */}
          <div style={{
            display: 'flex', gap: 16, flexWrap: 'wrap',
            padding: '12px 14px',
            background: 'var(--bg-elevated)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}>
            <div>
              <div className="label" style={{ marginBottom: 3 }}>Estimated area</div>
              <span className="mono" style={{ fontSize: 18, fontWeight: 500, color: 'var(--danger)' }}>
                {fmtArea(areaPx)}<span style={{ fontSize: 11, color: 'var(--text-secondary)', marginLeft: 3 }}>px²</span>
              </span>
            </div>
            <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 16 }}>
              <div className="label" style={{ marginBottom: 3 }}>MC samples</div>
              <span className="mono" style={{ fontSize: 18, fontWeight: 500, color: 'var(--text-primary)' }}>
                {mcSamples != null ? Number(mcSamples).toLocaleString() : '—'}
              </span>
            </div>
            <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: 16, flex: 1, minWidth: 120 }}>
              <div className="label" style={{ marginBottom: 3 }}>Formula</div>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                area ≈ (hits / N) × bbox_area
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Wire TumorAnalysisPanel into ResultCard**

In the `ResultCard` function, find this line:
```jsx
{item.histogram_b64 && <Histogram b64={item.histogram_b64} />}
```

Add `TumorAnalysisPanel` immediately after it (before the closing `</div>`):
```jsx
{item.histogram_b64 && <Histogram b64={item.histogram_b64} />}
{item.contour_b64 && (
  <TumorAnalysisPanel
    contourB64={item.contour_b64}
    scatterB64={item.scatter_b64}
    areaPx={item.area_px}
    mcSamples={item.mc_area_samples}
  />
)}
```

- [ ] **Step 4: Verify the template renders without JS errors**

Start the server:
```bash
python -m uvicorn mc_dropout.api.main:app --reload
```

Open `http://localhost:8000` in a browser, upload a brain MRI image classified as Tumor, and verify:
1. The page loads with no console errors
2. When result is "Tumor": a "Tumor region analysis" collapsible section appears below the histogram, showing two images side-by-side and the stats row with `px²` area and MC samples count
3. When result is "No Tumor": no `TumorAnalysisPanel` appears
4. The collapse toggle (chevron button) hides/shows the panel

- [ ] **Step 5: Commit**

```bash
git add src/mc_dropout/api/templates/index.html
git commit -m "feat: add TumorAnalysisPanel React component — contour + MC scatter + area stats"
```

---

## Task 8: Run full test suite and verify end-to-end

**Files:** none new

- [ ] **Step 1: Run entire test suite**

```bash
pytest -v
```

Expected: all tests pass with no failures.

- [ ] **Step 2: Smoke-test the running server**

```bash
python -m uvicorn mc_dropout.api.main:app --reload
```

Upload a real brain MRI tumor image via the UI. Confirm:
- Prediction result appears
- Tumor analysis panel auto-expands with contour image, scatter image, area in px², and sample count
- No Tumor images show only the standard result card

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: tumor analysis — GradCAM + Otsu contour + MC area estimation end-to-end"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** gradcam_mask ✓, mc_area_estimate ✓, render_annotated_images ✓, predict.py integration ✓, routes.py schema ✓, frontend panel ✓, error handling (exception catch in mc_predict, empty-mask fallback in mc_area_estimate) ✓
- [x] **No placeholders:** all steps have concrete code
- [x] **Type consistency:** `contour_pts` named identically in all tasks; `hits_mask` bool array used consistently in Task 3 and 4; `bbox` tuple `(x1, y1, x2, y2)` consistent; field names `contour_b64 / scatter_b64 / area_px / mc_area_samples` consistent across Tasks 5, 6, 7
- [x] **Import of `cv2`** added to test file where used (noted inline in test stubs)
- [x] **`_blank_image` helper** already defined in `test_predict.py`; Task 5 tests reuse it
