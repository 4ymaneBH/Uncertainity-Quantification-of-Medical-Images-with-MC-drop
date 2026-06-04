"""Post-processing comparison for contour completeness.

Localiser = largest shape-valid bright blob (the new, GradCAM-free approach).
For each audit image we build a 'lesion reference' by region-growing from the
selected blob at a lower threshold, then score each post-processing variant by
coverage / precision / Dice against that reference, plus intrinsic completeness
metrics (solidity, fragment count, filled holes) and a skull-safety check.

Out: audit_out/postproc/<name>/<variant>.png  and  audit_out/postproc/report.txt
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_out" / "postproc"
SAMPLES = ["y0.jpg", "y10.jpg", "y100.jpg", "y1000.jpg", "y1003.jpg"]
SZ = 150
BRIGHT_PCT = 85
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ── localiser: largest shape-valid bright blob ──────────────────────────────
def select_blob(gray: np.ndarray) -> np.ndarray | None:
    thr = float(np.percentile(gray, BRIGHT_PCT))
    _, bright = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, K3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bright)
    best, best_area = None, 0
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        if area < 30 or (bw >= SZ * 0.6 and bh >= SZ * 0.6):
            continue
        if area / (bw * bh) < 0.35:
            continue
        cm = (lab == i).astype(np.uint8)
        ct, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(ct, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(c))
        if (area / ha if ha > 0 else 0) < 0.5:
            continue
        if area > best_area:
            best_area, best = area, i
    if best is None:
        return None
    return (lab == best).astype(np.uint8) * 255


def roi_of(mask: np.ndarray, pad: int = 15):
    x, y, w, h = cv2.boundingRect(mask)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(SZ, x + w + pad), min(SZ, y + h + pad)
    return x1, y1, x2, y2


def lesion_reference(gray: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Region-grow from the blob at a lower (Otsu) threshold inside the ROI."""
    x1, y1, x2, y2 = roi_of(blob)
    ref = np.zeros_like(blob)
    sub = gray[y1:y2, x1:x2]
    t, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = (gray >= t).astype(np.uint8) * 255
    low[:y1] = 0; low[y2:] = 0; low[:, :x1] = 0; low[:, x2:] = 0
    n, lab = cv2.connectedComponents(low)
    touched = np.unique(lab[blob > 0])
    for lid in touched:
        if lid != 0:
            ref[lab == lid] = 255
    return fill(ref)


def fill(mask: np.ndarray) -> np.ndarray:
    ct, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if ct:
        cv2.drawContours(out, [max(ct, key=cv2.contourArea)], -1, 255, -1)
    return out


# ── post-processing variants (input: selected blob mask) ────────────────────
def v_baseline(blob, gray):
    return blob

def v_close(blob, gray):
    return cv2.morphologyEx(blob, cv2.MORPH_CLOSE, K5)

def v_close_open(blob, gray):
    m = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, K5)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, K3)

def v_dilate(blob, gray):
    return cv2.dilate(blob, K3, iterations=1)

def v_adaptive_roi(blob, gray):
    """Re-threshold (Otsu) inside the blob ROI, keep component touching blob, then close."""
    x1, y1, x2, y2 = roi_of(blob)
    out = np.zeros_like(blob)
    sub = gray[y1:y2, x1:x2]
    t, _ = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seg = (gray >= t).astype(np.uint8) * 255
    seg[:y1] = 0; seg[y2:] = 0; seg[:, :x1] = 0; seg[:, x2:] = 0
    n, lab = cv2.connectedComponents(seg)
    for lid in np.unique(lab[blob > 0]):
        if lid != 0:
            out[lab == lid] = 255
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, K5)

def v_merge_adjacent(blob, gray):
    """Merge bright components whose pixels lie within a dilated neighbourhood of the blob."""
    thr = float(np.percentile(gray, BRIGHT_PCT))
    _, bright = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    near = cv2.dilate(blob, K5, iterations=2)
    merged = cv2.bitwise_and(bright, near)
    merged = cv2.bitwise_or(merged, blob)
    return cv2.morphologyEx(merged, cv2.MORPH_CLOSE, K5)

def v_edge_assisted(blob, gray):
    """Close the blob, union with Canny-edge-enclosed region inside ROI."""
    x1, y1, x2, y2 = roi_of(blob)
    sub = gray[y1:y2, x1:x2]
    edges = cv2.Canny(sub, 40, 120)
    edges = cv2.dilate(edges, K3, iterations=1)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K5)
    filled = np.zeros_like(sub)
    ct, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, ct, -1, 255, -1)
    full = np.zeros_like(blob)
    full[y1:y2, x1:x2] = filled
    full = cv2.bitwise_and(full, cv2.dilate(blob, K5, iterations=3))  # keep near blob only
    out = cv2.bitwise_or(full, blob)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, K5)


VARIANTS = [
    ("baseline", v_baseline), ("close", v_close), ("close_open", v_close_open),
    ("dilate", v_dilate), ("adaptive_roi", v_adaptive_roi),
    ("merge_adjacent", v_merge_adjacent), ("edge_assisted", v_edge_assisted),
]


def metrics(mask, ref):
    n_frag, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    n_frag -= 1
    filled = fill(mask)
    raw_area = int((mask > 0).sum())
    fa = int((filled > 0).sum())
    holes = fa - raw_area
    ct, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sol = 0.0
    if ct:
        c = max(ct, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(c))
        sol = (cv2.contourArea(c) / ha) if ha > 0 else 0.0
    inter = int(((filled > 0) & (ref > 0)).sum())
    cov = inter / max(int((ref > 0).sum()), 1)          # recall vs lesion
    prec = inter / max(fa, 1)                            # 1 - skull/over spill
    dice = 2 * inter / max(fa + int((ref > 0).sum()), 1)
    return dict(area=fa, frags=n_frag, holes=max(holes, 0), sol=sol,
                cov=cov, prec=prec, dice=dice)


def overlay(gray, mask, ref):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rc, _ = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, rc, -1, (255, 80, 0), 1)       # reference = orange
    mc, _ = cv2.findContours(fill(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, mc, -1, (0, 255, 255), 1)      # variant = cyan
    return rgb


def main():
    ddir = ROOT / yaml.safe_load(open(ROOT / "config.yaml"))["dataset"]["dir"].lstrip("./") / "yes"
    OUT.mkdir(parents=True, exist_ok=True)
    agg = {v: [] for v, _ in VARIANTS}
    lines = []
    for name in SAMPLES:
        img = Image.open(ddir / name).convert("RGB").resize((SZ, SZ))
        gray = np.array(img.convert("L"))
        blob = select_blob(gray)
        d = OUT / Path(name).stem
        d.mkdir(parents=True, exist_ok=True)
        if blob is None:
            lines.append(f"{name}: NO BLOB"); continue
        ref = lesion_reference(gray, blob)
        Image.fromarray(cv2.cvtColor(ref, cv2.COLOR_GRAY2RGB)).save(d / "_reference.png")
        lines.append(f"\n### {name}  ref_area={int((ref>0).sum())}")
        lines.append(f"{'variant':>15} {'area':>5} {'frag':>4} {'hole':>5} {'sol':>5} {'cov':>5} {'prec':>5} {'dice':>5}")
        for vname, fn in VARIANTS:
            m = fn(blob.copy(), gray)
            mm = metrics(m, ref)
            agg[vname].append(mm)
            Image.fromarray(overlay(gray, m, ref)).save(d / f"{vname}.png")
            lines.append(f"{vname:>15} {mm['area']:>5} {mm['frags']:>4} {mm['holes']:>5} "
                         f"{mm['sol']:>5.2f} {mm['cov']:>5.2f} {mm['prec']:>5.2f} {mm['dice']:>5.2f}")

    lines.append(f"\n{'='*60}\n### MEANS over audit set")
    lines.append(f"{'variant':>15} {'sol':>5} {'cov':>5} {'prec':>5} {'dice':>5}")
    for v, _ in VARIANTS:
        rows = agg[v]
        mean = lambda k: sum(r[k] for r in rows) / len(rows)
        lines.append(f"{v:>15} {mean('sol'):>5.2f} {mean('cov'):>5.2f} {mean('prec'):>5.2f} {mean('dice'):>5.2f}")

    report = "\n".join(lines)
    (OUT / "report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
