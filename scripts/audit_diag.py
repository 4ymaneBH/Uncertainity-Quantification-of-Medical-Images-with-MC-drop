"""Quantitative diagnostics for the contour-localisation failure.

Prints, per image:
  * CAM flatness stats (max/mean/std, peak-to-mean ratio, fraction of frame in
    the 85th-pct attention region)
  * every connected-component candidate with the exact fields _pick_component_near
    scores on, plus which one wins and where the LARGEST bright blob sits.
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
from mc_dropout.tumor_analysis import _IMAGENET_MEAN, _IMAGENET_STD, _BRIGHT_PERCENTILE, _GRADCAM_PERCENTILE  # noqa: E402

SAMPLES = ["y0.jpg", "y10.jpg", "y100.jpg", "y1000.jpg", "y1003.jpg"]


def cam_for(image, model, device, image_size):
    tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                             transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])
    t = tf(image).unsqueeze(0).to(device)
    acts, grads = [], []
    fh = model.conv3.register_forward_hook(lambda m, i, o: acts.append(o))
    bh = model.conv3.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.eval(); model.zero_grad()
        out = model(t); out[0, 0].backward()
    finally:
        fh.remove(); bh.remove()
    a, g = acts[0].squeeze(0), grads[0].squeeze(0)
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((a * w).sum(0)).detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cv2.resize(cam, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    sz = cfg["dataset"]["image_size"]
    device = torch.device("cpu")
    model = CNNModel(dropout_rate=cfg["model"]["dropout_rate"], image_size=sz)
    model.load_state_dict(torch.load(ROOT / cfg["model"]["path"].lstrip("./"), map_location=device))
    model.to(device)
    ddir = ROOT / cfg["dataset"]["dir"].lstrip("./") / "yes"

    for name in SAMPLES:
        p = ddir / name
        if not p.exists():
            print(f"\n### {name}: MISSING"); continue
        img = Image.open(p).convert("RGB").resize((sz, sz))
        cam = cam_for(img, model, device, sz)
        peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
        att_thr = float(np.percentile(cam, _GRADCAM_PERCENTILE))
        att = (cam >= att_thr).astype(np.uint8) * 255

        gray = np.array(img.convert("L"))
        bthr = float(np.percentile(gray[att > 0] if (att > 0).any() else gray, _BRIGHT_PERCENTILE))
        _, bright = cv2.threshold(gray, bthr, 255, cv2.THRESH_BINARY)
        tumor_mask = cv2.bitwise_and(bright, att)

        print(f"\n{'='*78}\n### {name}  peak=(x={peak_x},y={peak_y})")
        print(f"CAM: max={cam.max():.3f} mean={cam.mean():.3f} std={cam.std():.3f} "
              f"peak/mean={cam.max()/max(cam.mean(),1e-9):.1f}x  "
              f"attention covers {100*(att>0).mean():.1f}% of frame")

        # largest bright blob overall (proxy for the visible tumor)
        nb, lb, sb, cb = cv2.connectedComponentsWithStats(bright)
        if nb > 1:
            big = 1 + int(np.argmax(sb[1:, cv2.CC_STAT_AREA]))
            print(f"LARGEST bright blob: area={sb[big,cv2.CC_STAT_AREA]} "
                  f"centroid=({cb[big,0]:.0f},{cb[big,1]:.0f}) "
                  f"dist_to_peak={np.hypot(cb[big,0]-peak_x, cb[big,1]-peak_y):.0f} "
                  f"(max_dist={sz*0.6:.0f})")

        # candidates in the bright∩attention mask, full scoring fields
        n, lab, st, cen = cv2.connectedComponentsWithStats(tumor_mask)
        rows = []
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            bw, bh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
            extent = area / (bw * bh) if bw * bh else 0
            cm = (lab == i).astype(np.uint8)
            ct, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 0.0
            if ct:
                c = max(ct, key=cv2.contourArea)
                ha = cv2.contourArea(cv2.convexHull(c))
                sol = area / ha if ha > 0 else 0
            cx, cy = bx + bw / 2, by + bh / 2
            dist = np.hypot(cx - peak_x, cy - peak_y)
            comp = min(sol, extent)
            score = dist / max(comp, 1e-3)
            passed = (area >= 30 and not (bw >= sz*0.6 and bh >= sz*0.6)
                      and extent >= 0.35 and sol >= 0.5 and dist < sz*0.6)
            rows.append((score if passed else 9e9, i, area, (bx, by, bw, bh),
                         round(sol, 2), round(extent, 2), round(dist, 0), round(score, 0), passed))
        rows.sort()
        print(f"{'id':>3} {'area':>5} {'bbox':>18} {'sol':>5} {'ext':>5} {'dist':>5} {'score':>7} pass")
        for _, i, area, bbox, sol, ext, dist, score, passed in rows:
            sel = "  <== SELECTED" if (passed and _ == rows[0][0] and rows[0][8]) else ""
            print(f"{i:>3} {area:>5} {str(bbox):>18} {sol:>5} {ext:>5} {dist:>5.0f} {score:>7.0f} {str(passed):>5}{sel}")


if __name__ == "__main__":
    main()
