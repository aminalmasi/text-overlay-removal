#!/usr/bin/env python3
"""Grow the mask along the carrier itself, until the carrier ends.

A solid or translucent carrier is a patch of near-uniform colour with a crisp edge, and
the mask usually lands *inside* it -- close enough to look right, close enough to leave a
rim behind after erasing. Everything tried so far inferred the carrier's extent from
character scores or stroke residuals, which describe the text, not the panel it sits on.

This works on the panel directly. For each mask component, the carrier's colour is taken
from the flat (non-stroke) pixels the mask already holds -- flat, because a glyph is not
the colour of the card it is printed on. The mask then grows outward one ring at a time
while the new pixels stay within tolerance of that colour, and stops the moment they do
not. On a photograph with no carrier there is no uniform colour to follow, so it stops
immediately -- which is what the clean-photo control is for.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parent   # deploy copy: this folder
ITEMS = ROOT / "experiments/grid12_v1/items_p61.json"
CLEAN_MAPS = ROOT / "experiments/mask_v2/cache/negative"

STROKE_RADIUS, FLAT = 8, 8.0
TOLERANCE, MAX_GROW, MIN_FLAT_SAMPLES = 14.0, 30, 60
EDGE_STOP = 18.0


def stroke_residual(grey, radius=STROKE_RADIUS):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, element).astype(np.float32)
    opened = cv2.morphologyEx(grey, cv2.MORPH_OPEN, element).astype(np.float32)
    value = grey.astype(np.float32)
    return np.maximum(closed - value, value - opened)


def flood(image, mask, tolerance=TOLERANCE, max_grow=MAX_GROW):
    """Grow along the carrier, per component, keeping only growth that stopped by itself.

    Colour similarity alone cannot separate a carrier from a flat photograph -- measured on
    86 clean photos it grew on 84 of them, more often than on the real ones. What separates
    them is that a carrier is *finite*: the flood reaches the panel's edge and terminates,
    while a wall or a table has no such edge and runs to the step limit. Growth still
    expanding at the limit is therefore evidence the surface was never a carrier, and is
    discarded.

    The test is per component. Applied to the whole frame it never fires: one long boundary
    that keeps finding a pixel keeps the global loop alive, and every other component's
    growth is thrown away with it.
    """
    rgb = np.asarray(image, np.float32)
    grey = np.asarray(image.convert("L"))
    flat = stroke_residual(grey) < FLAT
    blurred = cv2.GaussianBlur(grey, (0, 0), 1.2)
    gradient = cv2.magnitude(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    passable = gradient < EDGE_STOP

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask * 255).astype(np.uint8), connectivity=8)
    grown = mask.copy()
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    added, kept_components = 0, 0

    for index in range(1, count):
        left = max(0, stats[index, cv2.CC_STAT_LEFT] - max_grow - 2)
        top = max(0, stats[index, cv2.CC_STAT_TOP] - max_grow - 2)
        right = min(mask.shape[1], stats[index, cv2.CC_STAT_LEFT]
                    + stats[index, cv2.CC_STAT_WIDTH] + max_grow + 2)
        bottom = min(mask.shape[0], stats[index, cv2.CC_STAT_TOP]
                     + stats[index, cv2.CC_STAT_HEIGHT] + max_grow + 2)
        window = (slice(top, bottom), slice(left, right))
        component = (labels[window] == index)
        sample = component & flat[window]
        if sample.sum() < MIN_FLAT_SAMPLES:
            continue                     # only strokes here, no carrier surface to follow
        colour = np.median(rgb[window][sample].reshape(-1, 3), axis=0)
        local_rgb = rgb[window]
        near = np.linalg.norm(local_rgb - colour, axis=2) < tolerance
        allowed = near & passable[window] & ~mask[window]

        region = component.copy()
        converged, local_added = False, 0
        for _ in range(max_grow):
            frontier = (cv2.dilate(region.astype(np.uint8), cross) > 0) & ~region & allowed
            if not frontier.any():
                converged = True         # the surface ended, which is what a carrier does
                break
            region |= frontier
            local_added += int(frontier.sum())
        if not converged or local_added == 0:
            continue                     # unbounded: a photograph, not a panel
        grown[window] |= region & ~component
        added += local_added
        kept_components += 1
    return grown, added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="combo_w2")
    parser.add_argument("--name", default="combo_flood")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--max-grow", type=int, default=MAX_GROW)
    args = parser.parse_args()

    items = {str(i["id"]): i for i in json.loads(ITEMS.read_text())}
    out = HERE / "masks" / args.name
    out.mkdir(parents=True, exist_ok=True)
    before, after, grew = [], [], 0
    for path in sorted((HERE / "masks" / args.source).glob("*.png")):
        image_id = path.stem
        if image_id not in items:
            continue
        image = Image.open(items[image_id]["source"]).convert("RGB")
        mask = np.asarray(Image.open(path).convert("L")) > 127
        grown, added = flood(image, mask, args.tolerance, args.max_grow)
        Image.fromarray((grown * 255).astype(np.uint8), "L").save(out / path.name, compress_level=4)
        before.append(100 * float(mask.mean()))
        after.append(100 * float(grown.mean()))
        grew += int(added > 0)
    print(f"positives  {len(before)} masks   {st.median(before):.2f}%  ->  {st.median(after):.2f}%"
          f"   ({st.median(after) - st.median(before):+.2f}pp, grew on {grew}/{len(before)})")

    # the control: clean photographs, where there is no carrier to follow
    clean_before, clean_after, clean_grew = [], [], 0
    for seed_path in sorted(CLEAN_MAPS.glob("*_seed.png")):
        source = ROOT / "split_v1/train/0" / f"{seed_path.stem.replace('_seed','')}.jpg"
        if not source.exists():
            continue
        image = Image.open(source).convert("RGB")
        mask = np.asarray(Image.open(seed_path).convert("L")) > 127
        if not mask.any():
            continue
        grown, added = flood(image, mask, args.tolerance, args.max_grow)
        clean_before.append(100 * float(mask.mean()))
        clean_after.append(100 * float(grown.mean()))
        clean_grew += int(added > 0)
    if clean_before:
        print(f"clean      {len(clean_before)} photos   {st.median(clean_before):.2f}%  ->  "
              f"{st.median(clean_after):.2f}%   ({st.median(clean_after)-st.median(clean_before):+.2f}pp, "
              f"grew on {clean_grew}/{len(clean_before)})")


if __name__ == "__main__":
    main()
