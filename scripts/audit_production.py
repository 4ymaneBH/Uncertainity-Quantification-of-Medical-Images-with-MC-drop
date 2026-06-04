"""Full audit through the PRODUCTION code path (not the experimental script).

Imports the shipped `tumor_mask`, `mc_area_estimate`, `render_annotated_images`
from mc_dropout.tumor_analysis and runs them exactly as predict.py does.

Outputs per image:
  audit_out/production/<name>_after.png   contour overlay from production mask
  audit_out/production/<name>_seed.png    seed-vs-refined comparison
Reports area / centroid / contour stats, coverage vs lesion reference, and
verifies production reproduces the experimental adaptive_roi result.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mc_dropout.tumor_analysis import (  # noqa: E402
    tumor_mask, mc_area_estimate, render_annotated_images,
    _select_seed_blob, _solidity,
)

OUT = ROOT / "audit_out" / "production"
SAMPLES = ["y0.jpg", "y10.jpg", "y100.jpg", "y1000.jpg", "y1003.jpg"]
SZ = 150

# Expected coverage/dice from the experimental adaptive_roi run (audit_postproc).
EXPECTED_ADAPTIVE_AREA = {"y0": 2595, "y10": 932, "y100": 1731, "y1000": 1307, "y1003": 1863}
BASELINE_COV = {"y0": 0.66, "y10": 0.12, "y100": 1.00, "y1000": 1.00, "y1003": 0.49}


def lesion_reference(gray, seed):
    x, y, w, h = cv2.boundingRect(seed)
    pad = 15
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(SZ, x + w + pad), min(SZ, y + h + pad)
    t, _ = cv2.threshold(gray[y1:y2, x1:x2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = (gray >= t).astype(np.uint8) * 255
    low[:y1] = 0; low[y2:] = 0; low[:, :x1] = 0; low[:, x2:] = 0
    n, lab = cv2.connectedComponents(low)
    ref = np.zeros_like(seed)
    for lid in np.unique(lab[seed > 0]):
        if lid:
            ref[lab == lid] = 255
    return _fill(ref)


def _fill(mask):
    ct, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if ct:
        cv2.drawContours(out, [max(ct, key=cv2.contourArea)], -1, 255, -1)
    return out


def main():
    ddir = ROOT / yaml.safe_load(open(ROOT / "config.yaml"))["dataset"]["dir"].lstrip("./") / "yes"
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"{'img':>7} {'area':>6} {'centroid':>12} {'frags':>5} {'sol':>5} "
             f"{'perim':>6} {'cov':>5} {'cov_before':>10} {'reprod':>7}"]
    for name in SAMPLES:
        stem = Path(name).stem
        img = Image.open(ddir / name).convert("RGB").resize((SZ, SZ))
        gray = np.array(img.convert("L"))

        # ── production code path ───────────────────────────────────────────
        mask = tumor_mask(img, image_size=SZ)
        area_data = mc_area_estimate(mask)

        # stats from the production mask
        n_frag, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
        n_frag -= 1
        ct, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ct:
            c = max(ct, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = M["m10"] / M["m00"] if M["m00"] else 0
            cy = M["m01"] / M["m00"] if M["m00"] else 0
            cnt_area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            sol = _solidity((mask > 0).astype(np.uint8))
        else:
            cx = cy = cnt_area = perim = sol = 0

        # coverage vs lesion reference (same metric as the post-proc audit)
        seed = _select_seed_blob(gray)
        ref = lesion_reference(gray, seed) if seed is not None else np.zeros_like(gray)
        filled = _fill(mask)
        inter = int(((filled > 0) & (ref > 0)).sum())
        cov = inter / max(int((ref > 0).sum()), 1)

        # reproduction check vs experimental adaptive_roi area (filled)
        exp = EXPECTED_ADAPTIVE_AREA[stem]
        reprod = abs(int((filled > 0).sum()) - exp) <= max(40, 0.05 * exp)

        # ── overlays (production render + a manual after overlay) ───────────
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if ct:
            cv2.drawContours(rgb, [c], -1, (0, 255, 255), 1)
        Image.fromarray(rgb).save(OUT / f"{stem}_after.png")

        # seed (red) vs refined (cyan) to visualise the refinement gain
        cmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if seed is not None:
            sc, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(cmp, sc, -1, (255, 60, 0), 1)
        if ct:
            cv2.drawContours(cmp, [c], -1, (0, 255, 255), 1)
        Image.fromarray(cmp).save(OUT / f"{stem}_seed_vs_refined.png")

        lines.append(f"{stem:>7} {int(cnt_area):>6} {f'({cx:.0f},{cy:.0f})':>12} "
                     f"{n_frag:>5} {sol:>5.2f} {perim:>6.0f} {cov:>5.2f} "
                     f"{BASELINE_COV[stem]:>10.2f} {str(reprod):>7}")

    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
