#!/usr/bin/env python3
"""CRAFT to find the survivors, SAM2 to get their true extent -- inside the claimed region only.

Two things were wrong with the last repair mask, and they compound.

CRAFT's threshold is set for a detector that must not fire on tile ornament or tyre tread
anywhere in the frame, so it is deliberately mean: at 0.55 it claims three to five times
less area than at 0.20 on every failing image. But that caution is only needed *outside* the
region the first pass already repainted. Inside it, every pixel was going to be regenerated
anyway, so a false positive costs nothing that has not already been paid. Confining the
search to that region is what buys the right to lower the threshold, and both come from the
same idea: the first layer's own mask bounds the damage.

And a character body is not the whole mark. CRAFT scores the middle of a stroke, so a mask
built from its blobs covers the core of each letter and leaves the fringe -- which is why
repairs kept coming back with a fainter version of the same writing. SAM2 prompted with each
blob's box returns the extent of the thing that is actually there, which is the stroke and
its edge. Prompted with geometry alone it will happily segment the whole car, so its answer
is filtered by plausible area and clipped to the claimed region.

The filler is FLUX followed by LaMa over the same pixels, which measured best of the eight
tried: leftover blobs 60 -> 12 and invented 16 -> 0.
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

from pipeline import erase, masking                 # noqa: E402
from pipeline.run import lama_fill, seed_for        # noqa: E402
import detect_leftover as D                         # noqa: E402

ITEMS = HERE / "records/items_p62_flash38.json"   # overridable with --items
rect = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1,) * 2)
SAM_MIN, SAM_MAX = 0.10, 4.0     # of the prompt box; wider than the mask stage's filter
                                 # because here the box is one character, not one region


def eligible(region, mask_region, box_region, scope, level, outside):
    """Where a surviving blob is allowed to be found, which is what `scope` decides.

    The low threshold is not free. CRAFT at 0.25 claims three to five times the area it does
    at 0.55, and the licence to run it that low comes entirely from searching only pixels the
    first pass already repainted -- a false positive there costs nothing that has not already
    been paid. Inside the localizer's box but outside the mask is untouched photograph, and
    at 0.25 CRAFT will find writing in tile ornament and tyre tread. So `box` widens the
    search and keeps the cheap threshold, which is the risky reading of the idea, and
    `box-strict` widens it while putting the threshold back up outside the mask, which is the
    same idea with the bargain kept.
    """
    if scope == "mask":
        return (region > level) & mask_region
    if scope == "box":
        return (region > level) & box_region
    return (((region > level) & mask_region)
            | ((region > outside) & box_region & ~mask_region))


def survivors(hot, min_area):
    """Blobs in the eligible region. No run-length gate: the region is already bounded, so a
    lone surviving stroke counts."""
    hot = hot.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    out = []
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        out.append((labels == k, (int(x), int(y), int(x + w), int(y + h)),
                    int(stats[k, cv2.CC_STAT_AREA])))
    return out


def sam_extend(models, image, boxes, shape):
    """SAM2's outline for each character box, filtered by plausible area."""
    if not boxes:
        return np.zeros(shape, bool)
    grown = np.zeros(shape, bool)
    shapes = masking.sam_shapes(models["sam"], models["processor"], image, boxes)
    for box, got in zip(boxes, shapes):
        if got is None:
            continue
        area = (got > 127).sum()
        box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        if not (SAM_MIN * box_area <= area <= SAM_MAX * box_area):
            continue
        grown |= got > 127
    return grown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="n_h10_sub")
    parser.add_argument("--level", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--grow", type=int, default=4)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--scope", choices=["mask", "box", "box-strict"], default="mask",
                        help="where to look for survivors: the first layer's mask, the "
                             "localizer's whole box, or the box with the threshold put back "
                             "up outside the mask")
    parser.add_argument("--outside", type=float, default=0.55,
                        help="CRAFT's own threshold, used outside the mask under box-strict")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--guard-rule", choices=["boxes", "maskfix"], default="boxes",
                        help="boxes: put the photograph back inside every protected box (the original "
                             "behaviour). maskfix: the first layer's rule as well -- a protected surface "
                             "over 10%% of the frame shields only overlays at least 80%% inside it -- so an "
                             "overlay line lying across a phone screen, erased by the first layer, is not "
                             "restored here (s010)")
    parser.add_argument("--items", type=Path, default=ITEMS)
    parser.add_argument("--core", default=None,
                        help="core mask dir; defaults to the arm's own")
    args = parser.parse_args()
    globals()["ITEMS"] = args.items
    if args.core:
        globals()["CORE"] = HERE / "masks" / args.core
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
            first = HERE / "results" / args.base / f"{image_id}.png"
            mask_p = HERE / "masks" / args.base / f"{image_id}.png"
            if not first.exists() or not mask_p.exists():
                continue
            item = items[image_id]
            source = np.asarray(Image.open(item["source"]).convert("RGB"))
            current = Image.open(first).convert("RGB")
            mask_region = np.asarray(Image.open(mask_p).convert("L")) > 127
            box_region = np.zeros(mask_region.shape, bool)
            for region_ in item["boxes"]:
                if region_.get("physically_in_scene"):
                    continue
                left, top, right, bottom = masking.widen(
                    masking.to_pixels(region_["box_2d"], current.size), current.size)
                box_region[max(0, top):bottom, max(0, left):right] = True
            claimed = mask_region if args.scope == "mask" else box_region
            guard = np.zeros(source.shape[:2], bool)
            for region_ in item["boxes"]:
                if not region_.get("physically_in_scene"):
                    continue
                box = region_.get("text_box_2d") or region_["box_2d"]
                left, top, right, bottom = masking.to_pixels(box, current.size)
                guard[max(0, top):bottom, max(0, left):right] = True
            if args.guard_rule == "maskfix":
                from run_maskfix import guard_mask
                guard &= guard_mask(item, current.size)

            started = time.perf_counter()
            region, _ = D.region_score(craft, np.asarray(current))
            found = survivors(eligible(region, mask_region, box_region, args.scope,
                                       args.level, args.outside), args.min_area)
            repair = np.zeros(source.shape[:2], bool)
            for blob, _, _ in found:
                repair |= blob
            n_sam = 0
            if found and not args.no_sam:
                extended = sam_extend(models, current, [b for _, b, _ in found],
                                      source.shape[:2])
                n_sam = int(extended.sum())
                repair |= extended & claimed          # never outside what we already took
            if args.grow and repair.any():
                repair = cv2.dilate((repair * 255).astype(np.uint8),
                                    rect(args.grow)) > 127
            repair &= claimed
            repair &= ~guard

            if repair.any():
                mask = Image.fromarray((repair * 255).astype(np.uint8), "L")
                mask.save(mask_dir / f"{image_id}.png")
                seed = seed_for(image_id) + 104729
                current = erase.run(models["flux"], models["embeds"], current, mask, seed=seed)
                current = lama_fill(models["lama"], current, mask)
                # the same guarantee as the first layer: the feather in erase.run reaches
                # past the mask edge, so a protected box beside a repair can still be
                # altered. Put the photograph back inside every guard box, always.
                guard_original = np.asarray(Image.open(item["source"]).convert("RGB"),
                                            np.uint8)
                pixels = np.asarray(current, np.uint8).copy()
                pixels[guard] = guard_original[guard]
                current = Image.fromarray(pixels, "RGB")
            current.save(out_dir / f"{image_id}.png")
            handle.write(json.dumps({
                "id": image_id, "tag": args.tag, "blobs": len(found),
                "repair_px": int(repair.sum()), "sam_px": n_sam,
                "repaired": bool(repair.any()),
                "seconds": round(time.perf_counter() - started, 2)}) + "\n")
    print(json.dumps({"finished": args.tag}), flush=True)


if __name__ == "__main__":
    main()
