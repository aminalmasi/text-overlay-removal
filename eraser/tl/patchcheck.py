#!/usr/bin/env python3
"""Does a repainted hole show as a patch? Two measures per hole of the stencil, only where the
photograph around the hole is smooth (sky, wall, plain backdrop) -- elsewhere a hole that continues
real texture or spans two backgrounds differs from its surroundings for good reasons:

  grain  fine-texture energy inside the hole over the energy just outside it: the speckled
         rectangle on s032 is 2-4x, a smooth repaint of smooth sky about 1x
  seam   the colour step across the hole's edge, pixel against nearest pixel (CIELAB distance, 0-100
         scale), median along the edge: a tone patch shows a step all round (s045, s060: ~4),
         a clean repaint continues its neighbours (< 2)

found on the compare_best review (2026-10-11): the noise patches on s032, s045 and s060.
"""
from __future__ import annotations

import cv2
import numpy as np

SMOOTH_MAX = 2.0        # mean fine texture around the hole (L channel, 0-255) below which it is smooth
AREA_MAX = 0.05         # holes over 5% of the frame are content, not a watermark-sized patch
GRAIN_MIN = 2.0
SEAM_MIN = 2.5


def _k(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


def _hf(L):
    return np.abs(L - cv2.GaussianBlur(L, (0, 0), 2.0))


def holes(source_bgr, repaint_bgr, stencil):
    """One row per hole of the stencil: its mask and the measures. stencil: bool HxW."""
    O = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    R = cv2.cvtColor(repaint_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hO, hR = _hf(O[..., 0]), _hf(R[..., 0])
    n, lab, st, _ = cv2.connectedComponentsWithStats(stencil.astype(np.uint8), 8)
    din = cv2.distanceTransform(stencil.astype(np.uint8), cv2.DIST_L2, 5)
    dout = cv2.distanceTransform((~stencil).astype(np.uint8), cv2.DIST_L2, 5)
    rows = []
    for c in range(1, n):
        area = st[c, cv2.CC_STAT_AREA] / stencil.size
        if area < 0.0005:
            continue
        hole = lab == c
        near = cv2.dilate(hole.astype(np.uint8), _k(14)) > 0
        ring = near & ~stencil & (dout >= 4)
        inner = hole & (din >= 4)
        if inner.sum() < 50 or ring.sum() < 200:
            continue
        smooth = float(hO[ring].mean())
        grain = float(hR[inner].mean() / max(smooth, 0.3))
        band_in = hole & (din >= 3) & (din <= 5)
        band_out = near & ~stencil & (dout >= 3) & (dout <= 5)
        seam = 0.0
        if band_in.sum() >= 40 and band_out.sum() >= 40:
            _, lin = cv2.distanceTransformWithLabels((~band_in).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
            ys, xs = np.nonzero(band_in)
            idx = np.full(lin.max() + 1, -1, int)
            idx[lin[ys, xs]] = np.arange(len(ys))
            oy, ox = np.nonzero(band_out)
            j = idx[lin[oy, ox]]
            ok = j >= 0
            d = np.linalg.norm(R[ys[j[ok]], xs[j[ok]]] - O[oy[ok], ox[ok]], axis=1) * 100 / 255
            seam = float(np.median(d)) if d.size else 0.0
        rows.append({"mask": hole, "area_pct": round(100 * area, 2), "smooth": round(smooth, 2),
                     "grain": round(grain, 2), "seam": round(seam, 2)})
    return rows


def flagged(row):
    return (row["smooth"] < SMOOTH_MAX and row["area_pct"] < 100 * AREA_MAX
            and (row["grain"] >= GRAIN_MIN or row["seam"] >= SEAM_MIN))


def badness(row):
    """1.0 at the flag threshold; lower is better."""
    return max(row["grain"] / GRAIN_MIN, row["seam"] / SEAM_MIN)
