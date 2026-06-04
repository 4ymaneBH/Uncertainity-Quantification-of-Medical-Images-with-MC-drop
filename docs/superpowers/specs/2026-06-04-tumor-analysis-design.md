# Tumor Analysis: Contour Detection & Monte Carlo Area Estimation

**Date:** 2026-06-04  
**Status:** Approved

## Goal

Extend the existing MC Dropout classifier with a tumor analysis pipeline that, when the prediction is "Tumor":
1. Localises the tumor region using GradCAM + Otsu thresholding
2. Draws the contour with a stroke on a zoomed crop
3. Estimates the tumor area in pixels using Monte Carlo integration
4. Returns two annotated images (clean contour + MC scatter) alongside the area stat

This emphasises Monte Carlo's utility for estimating areas under complex boundaries — a natural complement to the uncertainty-quantification theme of the project.

## Architecture

### New file: `src/mc_dropout/tumor_analysis.py`

Three focused, independently-testable functions:

**`gradcam_mask(image, model, device, image_size) → np.ndarray`**
- Registers a forward hook on `model.conv3` and a backward hook to capture gradients
- Runs one forward pass, backpropagates on the Tumor logit
- Computes gradient-weighted average of the conv3 feature maps (GradCAM formula)
- Upsamples heatmap to `image_size × image_size` with bilinear interpolation
- Thresholds at the 75th percentile → coarse attention mask
- Runs Otsu's threshold (via `cv2.threshold` with `THRESH_OTSU`) on the grayscale image *inside* the attention mask
- Returns the final clean binary mask

**`mc_area_estimate(mask, n_samples=50_000) → dict`**
- Finds the largest external contour from the mask (OpenCV `findContours`)
- Computes a tight bounding box with 20px padding
- Throws `n_samples` uniformly-random (x, y) points into the bounding box
- Tests each point against the contour polygon with `cv2.pointPolygonTest`
- Returns: `area_px` (float), `mc_samples_used` (int), `hits` (int), `points_xy` (ndarray), `hits_mask` (bool array), `contour_pts` (ndarray)

**`render_annotated_images(original_image, contour_pts, bbox, points_xy, hits_mask) → (str, str)`**
- Crops the original PIL image to `bbox` (with padding)
- **Contour image**: draws the contour as a 2px cyan stroke on the crop → base64 PNG
- **Scatter image**: same crop + contour stroke + green dots (hits) and red dots (misses), alpha-blended for legibility → base64 PNG
- Returns `(contour_b64, scatter_b64)`

### Changes to existing files

**`src/mc_dropout/predict.py`**
- After computing `mean_prob`/`std_dev`, if `prediction == "Tumor"`:
  - Call `gradcam_mask()` → mask
  - Call `mc_area_estimate(mask)` → area data
  - Call `render_annotated_images(...)` → two b64 strings
  - Merge into return dict as optional fields
- When prediction is "No Tumor" all four fields are `None`

**`src/mc_dropout/api/routes.py`**
- `PredictionResponse` gains 4 optional fields:
  ```python
  contour_b64: Optional[str] = None
  scatter_b64: Optional[str] = None
  area_px: Optional[float] = None
  mc_area_samples: Optional[int] = None
  ```
- `BatchPredictionItem` gets the same 4 optional fields

**`src/mc_dropout/api/templates/index.html`**
- New `TumorAnalysisPanel` React component rendered when `result.contour_b64` is truthy
- Auto-expands below the existing result card (histogram + probability gauge)
- Layout: side-by-side images (contour left, scatter right) + stat row below (`area_px` in px² + `mc_area_samples` count + formula `area ≈ (hits/N) × bbox_area`)

## Data Flow

```
Upload image
    ↓
mc_predict()
    ├── 100 MC Dropout forward passes → mean_prob, std_dev, histogram
    └── if Tumor:
            ├── gradcam_mask()  →  binary mask (GradCAM + Otsu)
            ├── mc_area_estimate(mask, n=50_000)  →  area_px, points
            └── render_annotated_images()  →  contour_b64, scatter_b64
    ↓
PredictionResponse (8 fields total, 4 optional)
    ↓
Frontend: existing result card + TumorAnalysisPanel (auto-shown when Tumor)
```

## Error Handling

- If GradCAM produces no usable mask (e.g., no bright region above threshold): fall back to full-image bounding box, log a warning, still return area estimate
- If no contour found after Otsu: return `area_px=0`, `contour_b64=None`, `scatter_b64=None` — frontend hides the panel
- Tumor analysis exceptions are caught in `mc_predict()` and cause all 4 fields to be `None` (prediction result is unaffected)

## Dependencies

- `opencv-python` (`cv2`) — already a likely transitive dep; must be added to `pyproject.toml` / `requirements.txt` if missing
- No new model weights required

## Key Numbers

| Parameter | Value | Rationale |
|---|---|---|
| GradCAM threshold percentile | 75th | Captures top-quarter activation; tunable |
| MC area samples | 50,000 | Fast (~10ms), accurate to ±0.5% on typical tumor shapes |
| Contour stroke width | 2px | Visible without obscuring detail |
| Zoom padding | 20px | Context around tight bounding box |
| Scatter dot radius | 1px | Legible at typical image sizes without overlap |
| Scatter dots displayed | min(5000, n_samples) | Cap render points to keep image light |
