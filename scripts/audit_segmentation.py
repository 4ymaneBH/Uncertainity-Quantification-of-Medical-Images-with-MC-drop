"""Segmentation pipeline audit.

Runs the tumor-localisation pipeline on a handful of real tumour MRIs and dumps
every intermediate stage to disk so we can see *exactly* where the contour
stops tracking the tumour:

    stage 0  original MRI (resized to model input size)
    stage 1  GradCAM "probability"/attention heatmap over the MRI
    stage 2  brightness binary mask (prob>thresh analogue)
    stage 3  attention-restricted mask (bright AND attention)
    stage 4  largest-connected-component mask actually used for the contour
    stage 5  contour overlay on the MRI

Run:  python scripts/audit_segmentation.py
Out:  audit_out/<name>/stage*.png  and audit_out/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mc_dropout.model import CNNModel  # noqa: E402
from mc_dropout.tumor_analysis import (  # noqa: E402
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _BRIGHT_PERCENTILE,
    _GRADCAM_PERCENTILE,
    _pick_component_near,
)

OUT = ROOT / "audit_out"
SAMPLES = ["y0.jpg", "y10.jpg", "y100.jpg", "y1000.jpg", "y1003.jpg"]


def _heatmap_overlay(gray_rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blend a 0..1 cam over an RGB image as a JET heatmap."""
    cam_u8 = np.uint8(255 * cam)
    hm = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(gray_rgb, 0.55, hm, 0.45, 0)


def _save(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def audit_one(name: str, img_path: Path, model: CNNModel, device, image_size: int) -> str:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    orig_dims = image.size  # (W, H)
    resized = image.resize((image_size, image_size))
    rgb = np.array(resized)

    # ── stage 0 : original ──────────────────────────────────────────────
    _save(out_dir / "stage0_original.png", rgb)

    # ── GradCAM (the only spatial "probability" this model can produce) ──
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = transform(resized).unsqueeze(0).to(device)

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    fh = model.conv3.register_forward_hook(lambda _m, _i, o: activations.append(o))
    bh = model.conv3.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.eval()
        model.zero_grad()
        out = model(tensor)
        prob = float(out.item())
        out[0, 0].backward()
    finally:
        fh.remove()
        bh.remove()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    cam_native_hw = tuple(act.shape[1:])  # (H, W) of conv3 feature map
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((act * weights).sum(dim=0)).detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_resized = cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

    # ── stage 1 : heatmap over MRI ──────────────────────────────────────
    overlay = _heatmap_overlay(rgb, cam_resized)
    cv2.circle(overlay, (int(peak_x), int(peak_y)), 4, (255, 255, 255), 1)
    _save(out_dir / "stage1_gradcam_heatmap.png", overlay)

    # ── stage 2 : brightness mask ───────────────────────────────────────
    gray = np.array(resized.convert("L"))
    cam_thresh = float(np.percentile(cam_resized, _GRADCAM_PERCENTILE))
    attention_mask = (cam_resized >= cam_thresh).astype(np.uint8) * 255
    attended = gray[attention_mask > 0]
    bright_thresh = float(np.percentile(attended if attended.size else gray, _BRIGHT_PERCENTILE))
    _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    _save(out_dir / "stage2_bright_mask.png", cv2.cvtColor(bright_mask, cv2.COLOR_GRAY2RGB))

    # ── stage 3 : bright AND attention ──────────────────────────────────
    tumor_mask = cv2.bitwise_and(bright_mask, attention_mask)
    _save(out_dir / "stage3_bright_and_attention.png", cv2.cvtColor(tumor_mask, cv2.COLOR_GRAY2RGB))

    # ── stage 4 : largest/nearest connected component ───────────────────
    cc = _pick_component_near(tumor_mask, int(peak_x), int(peak_y))
    cc_mask = cc if cc is not None else np.zeros_like(tumor_mask)
    _save(out_dir / "stage4_component_mask.png", cv2.cvtColor(cc_mask, cv2.COLOR_GRAY2RGB))

    # ── stage 5 : contour overlay ───────────────────────────────────────
    contour_overlay = rgb.copy()
    contours, _ = cv2.findContours(cc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_area = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(c)
        cv2.drawContours(contour_overlay, [c], -1, (0, 255, 255), 1)
    cv2.circle(contour_overlay, (int(peak_x), int(peak_y)), 4, (255, 0, 0), 1)
    _save(out_dir / "stage5_contour_overlay.png", contour_overlay)

    return (
        f"{name}: orig={orig_dims} resized=({image_size},{image_size}) "
        f"cam_native_hw={cam_native_hw} prob={prob:.3f} peak=(x={peak_x},y={peak_y}) "
        f"bright_px={int((bright_mask>0).sum())} tumor_px={int((tumor_mask>0).sum())} "
        f"cc_found={cc is not None} contour_area_px={cnt_area:.0f}"
    )


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    image_size = cfg["dataset"]["image_size"]
    weights = ROOT / cfg["model"]["path"].lstrip("./")

    device = torch.device("cpu")
    model = CNNModel(dropout_rate=cfg["model"]["dropout_rate"], image_size=image_size)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    OUT.mkdir(exist_ok=True)
    lines = []
    data_dir = ROOT / cfg["dataset"]["dir"].lstrip("./") / "yes"
    for name in SAMPLES:
        p = data_dir / name
        if not p.exists():
            lines.append(f"{name}: MISSING")
            continue
        lines.append(audit_one(Path(name).stem, p, model, device, image_size))

    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
