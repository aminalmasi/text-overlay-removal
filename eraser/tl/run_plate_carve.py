#!/usr/bin/env python3
"""Take the overlay off a protected object without inventing what the object says.

A protected region -- a plate, a badge, print on the goods -- is subtracted from the mask as
a rectangle, so an overlay word that happens to cross one is protected along with it. On
9625907 an orange caption runs straight over a licence plate, and the word sitting on the
digits survives both layers because every pixel of the plate was taken out of the mask.

Erasing it is not simply a matter of putting those pixels back. Whatever fills them is
inventing part of a licence number, which is the one thing the protection exists to prevent.
So this pass is narrow on both axes:

  what it takes   only pixels that CRAFT scores as a character body and that belong to a
                  run of writing which *continues outside* the protected box. A caption
                  crossing a plate is one run passing over it; the plate's own digits are a
                  run that begins and ends inside it, so they are never selected. Colour
                  agreement with the caption's ink, sampled where it runs in the clear, is
                  required as well -- on its own it was not enough, since a loose tolerance
                  reaches from orange caption to black digits.
  what fills it   LaMa, never FLUX. LaMa continues the texture around a thin stroke; it has
                  no capacity to write a plausible digit, which here is a feature.

If the ink cannot be sampled, or the selection would take more than a quarter of the
protected region, the pass declines and the image is copied through unchanged. Three of the
90 overlay photographs have a box overlapping a protected region at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent   # deploy copy: this folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/added_evidence_v1"))
HERE = ROOT / "experiments/creativity_v1"
sys.path.insert(0, str(HERE))

from pipeline import masking                        # noqa: E402
from pipeline.run import lama_fill                  # noqa: E402
import detect_leftover as D                         # noqa: E402

INK, MIN_SAMPLE, MIN_BLOB, GROW, CEILING = 0.55, 200, 20, 2, 0.25
TOLERANCE, OUTSIDE = 90.0, 150   # colour distance cap; px of a run that must lie in the clear
rect = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1,) * 2)


def regions(item, size, scene):
    canvas = np.zeros((size[1], size[0]), bool)
    for region in item["boxes"]:
        if bool(region.get("physically_in_scene")) != scene:
            continue
        box = masking.to_pixels(region["box_2d"], size)
        if not scene:
            box = masking.widen(box, size)
        left, top, right, bottom = box
        canvas[max(0, top):bottom, max(0, left):right] = True
    return canvas


def carve(source, score, affinity, erase_area, protect):
    """The overlay's own strokes where they lie on a protected object, or None."""
    overlap = erase_area & protect
    if not overlap.any():
        return None, "no overlap"
    ink = (score > INK) & erase_area & ~protect
    if int(ink.sum()) < MIN_SAMPLE:
        return None, "no ink to sample"
    sample = source[ink].astype(np.float32)
    middle = np.median(sample, axis=0)
    spread = np.median(np.abs(sample - middle), axis=0).mean()
    tolerance = min(TOLERANCE, max(40.0, 3.0 * float(spread)))
    close = np.linalg.norm(source.astype(np.float32) - middle, axis=2) < tolerance
    # A caption crossing a plate is one run of writing that passes over it. The plate's own
    # printing is a run that starts and finishes inside the protected box. Joining character
    # bodies through CRAFT's affinity map and asking which runs have mass in the clear is
    # what tells the two apart -- colour alone reaches from an orange caption to a black
    # digit at any tolerance loose enough to catch the caption's own anti-aliased edge.
    joined = ((score > INK) | (affinity > D.AFFINITY)).astype(np.uint8)
    count, runs = cv2.connectedComponents(joined, 8)
    clear = erase_area & ~protect
    crossing = np.zeros_like(overlap)
    for k in range(1, count):
        run = runs == k
        if not (run & overlap).any():
            continue
        if int((run & clear).sum()) >= OUTSIDE:
            crossing |= run
    if not crossing.any():
        return None, "no run of writing crosses into the protected region"
    picked = (score > INK) & overlap & close & crossing
    if not picked.any():
        return None, "no matching ink on the protected region"
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        picked.astype(np.uint8), 8)
    keep = np.zeros_like(picked)
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] >= MIN_BLOB:
            keep |= labels == k
    if not keep.any():
        return None, "only speckle"
    keep = cv2.dilate((keep * 255).astype(np.uint8), rect(GROW)) > 127
    keep &= protect
    if keep.sum() > CEILING * protect.sum():
        return None, f"would take {100 * keep.sum() / protect.sum():.0f}% of the protection"
    return keep, f"tolerance {tolerance:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--items", type=Path,
                        default=HERE / "records/items_p62_flash38.json")
    args = parser.parse_args()
    items = {str(i["id"]): i for i in json.loads(args.items.read_text())}

    from pipeline import load_all
    from craft_model import load_craft
    models = load_all()
    craft = load_craft(D.CRAFT_W, "cuda")
    out_dir, mask_dir = HERE / "results" / args.tag, HERE / "masks" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = HERE / f"records/{args.tag}.jsonl"

    with records.open("a", encoding="utf-8", buffering=1) as handle:
        for image_id in sorted(items):
            if (out_dir / f"{image_id}.png").exists():
                continue
            base = HERE / "results" / args.base / f"{image_id}.png"
            if not base.exists():
                continue
            item = items[image_id]
            current = Image.open(base).convert("RGB")
            source = np.asarray(Image.open(item["source"]).convert("RGB"))
            if source.shape[:2] != (current.height, current.width):
                current.save(out_dir / f"{image_id}.png")
                continue
            protect = regions(item, current.size, True)
            erase_area = regions(item, current.size, False)
            if not (protect & erase_area).any():
                current.save(out_dir / f"{image_id}.png")
                continue
            started = time.perf_counter()
            score, affinity = D.region_score(craft, source)
            picked, why = carve(source, score, affinity, erase_area, protect)
            if picked is None:
                current.save(out_dir / f"{image_id}.png")
                row = {"id": image_id, "tag": args.tag, "carved": False, "why": why}
            else:
                mask = Image.fromarray((picked * 255).astype(np.uint8), "L")
                mask.save(mask_dir / f"{image_id}.png")
                lama_fill(models["lama"], current, mask).save(out_dir / f"{image_id}.png")
                row = {"id": image_id, "tag": args.tag, "carved": True,
                       "px": int(picked.sum()),
                       "share_of_protection": round(float(picked.sum() / protect.sum()), 3),
                       "why": why, "seconds": round(time.perf_counter() - started, 2)}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    print(json.dumps({"finished": args.tag}), flush=True)


if __name__ == "__main__":
    main()
