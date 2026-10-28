#!/usr/bin/env python3
"""The same eraser, with the voids inside a text box filled in before the halo goes on.

The evidence mask is built from SAM2's outline unioned with what a throwaway repaint moved.
Neither signal is reliable on a dark background, so on a line of white text lying across a
tyre or a dark cabinet the mask comes back with a hole in the middle of the writing -- 13 to
36 px of unclaimed pixels sitting wholly inside a box the localizer got right. FLUX is then
asked to repaint a line with a gap in it, and the letters in the gap survive untouched.

The fix uses the one thing a text line guarantees: evidence on both sides of the void. Within
each box, a gap between two claimed pixels on the same row is closed -- but only if it is
short enough to be a gap rather than a distance. Bridging every row end to end also joins the
two lines of a two-line caption through the blank strip between them, and then the eraser is
handed a solid slab where it had a pair of ribbons; on 9827612 that cost the bottom lip of the
machine being photographed. Capping the bridge at a tenth of the long side keeps every one of
the four survivors found by eye and gives back three quarters of the extra area: +0.19 points
of mask over the 90 overlay photographs, against +0.36 uncapped and +1.64 for simply taking
the whole box.
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
HERE = ROOT / "experiments/creativity_v1"

from pipeline import config as C                    # noqa: E402
from pipeline import erase, masking                 # noqa: E402
from pipeline.run import seed_for                   # noqa: E402
from flood_carrier import flood                     # noqa: E402

CORE = HERE / "masks/core_p62"
HALO = 10
BRIDGE = 0.10        # longest gap worth closing, as a fraction of the frame's long side
rect = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1,) * 2)


def boxes_of(item, size, scene):
    """Erase boxes are widened the way the mask stage widens them; protected ones are not."""
    out = []
    for region in item["boxes"]:
        if bool(region.get("physically_in_scene")) != scene:
            continue
        box = masking.to_pixels(region["box_2d"], size)
        if not scene:
            box = masking.widen(box, size)
        left, top, right, bottom = box
        if right > left and bottom > top:
            out.append((max(0, left), max(0, top), right, bottom))
    return out


SOFT_ABOVE = 10.0      # % of frame; above this a protected region is a surface, not a label
MOSTLY_IN = 0.80       # an overlay this far inside a soft region is left alone


def guard_mask(item, size):
    """Which pixels may not be touched -- and a licence plate is not a phone screen.

    Both come back as `physically_in_scene`, and the pipeline has treated them identically:
    every pixel inside either is copied back from the original after every pass. On a plate
    that is exactly right and it is not negotiable. On a screen it is how 9914389 kept its
    overlay -- the localizer declared the whole handset display in scene, 51% of the frame,
    and a speech bubble lying across the top of it was therefore untouchable.

    Size tells them apart with room to spare. In this set the plates and labels run 0.0% to
    5.0% of the frame and the three screens run 43.8%, 51.2% and 68.3%; nothing lies between.
    So a small protected region stays absolute, and a large one shields an overlay only when
    the overlay is genuinely *on* it -- four fifths inside. An overlay that merely clips its
    edge is erased in full, screen included, because half an erased bubble is worse than
    either a whole one or none.
    """
    guard = np.zeros((size[1], size[0]), bool)
    soft = []
    for region in item["boxes"]:
        if not region.get("physically_in_scene"):
            continue
        left, top, right, bottom = masking.to_pixels(region["box_2d"], size)
        left, top = max(0, left), max(0, top)
        if right <= left or bottom <= top:
            continue
        guard[top:bottom, left:right] = True
        if 100 * (right - left) * (bottom - top) / (size[0] * size[1]) > SOFT_ABOVE:
            soft.append((left, top, right, bottom))
    if not soft:
        return guard
    for left, top, right, bottom in boxes_of(item, size, False):
        area = max(1, (right - left) * (bottom - top))
        inside = max(((min(right, sr) - max(left, sl)) * (min(bottom, sb) - max(top, st))
                      for sl, st, sr, sb in soft
                      if min(right, sr) > max(left, sl) and min(bottom, sb) > max(top, st)),
                     default=0)
        if 0 < inside / area < MOSTLY_IN:
            guard[max(0, top):bottom, max(0, left):right] = False
    return guard


def keep_protected(original, result, guard):
    """Put the photograph back wherever the localizer said it belongs to the scene.

    Zeroing the guard out of the mask is not enough on its own. The composite feathers the
    mask with a Gaussian of sigma 8, so the blend reaches roughly 24 px past the mask's edge
    -- and where a mask runs up against a protected box, that reach crosses the boundary.
    Measured on the arms that actually had protected regions, six images had pixels altered
    inside one, up to 65% of the plate on 9625907 and a worst-case error of 135/255.

    A licence plate is not a thing to be nearly right about, so this is enforced rather than
    argued: after the repaint, every pixel inside a protected box is taken from the original.
    The mask cannot reach it, the feather cannot reach it, and no future change to either can
    reintroduce the leak.
    """
    if not guard.any():
        return result, 0
    a = np.asarray(original, np.uint8)
    b = np.asarray(result, np.uint8).copy()
    moved = int((np.abs(b.astype(np.int16) - a.astype(np.int16)).max(axis=2) > 2)[guard].sum())
    b[guard] = a[guard]
    return Image.fromarray(b, "RGB"), moved


def close_rows(core, boxes, cap):
    """Close the short gaps in each row of each box; leave the long ones open."""
    filled = core.copy()
    for left, top, right, bottom in boxes:
        window = core[top:bottom, left:right]
        if not window.any():
            continue
        patch = window.copy()
        for y in range(window.shape[0]):
            hit = np.nonzero(window[y])[0]
            if hit.size < 2:
                continue
            for start, gap in zip(hit[:-1], np.diff(hit)):
                if 1 < gap <= cap + 1:
                    patch[y, start:start + gap] = True
        filled[top:bottom, left:right] |= patch
    return filled


def draw_best(models, image, mask, seed, draws, region):
    """Repaint several times and keep the least idiosyncratic result.

    A large hole is not one answer with noise on it; it is a sample. On 9796889 four draws of
    the identical mask gave a ghost, a clean floor, a cat and a waste bin -- one of them right
    and three of them inventing an object that was never there. Nothing in the image tells
    you which is which, but the draws tell you about each other: whatever is genuinely implied
    by the surroundings appears in all of them, and an invention appears in one.

    So draw N, take the pixel-wise median inside the mask as a picture of what the draws
    agree on, and keep the actual draw that sits closest to it. The median itself is not used
    as the output -- averaging four samples blurs, and the point is to keep a real one. It is
    the outlier that gets dropped, which is exactly where the cat lives.
    """
    if draws <= 1:
        return erase.run(models["flux"], models["embeds"], image, mask, seed=seed), 0, 0.0
    candidates = [erase.run(models["flux"], models["embeds"], image, mask,
                            seed=seed + n * 7919) for n in range(draws)]
    stack = np.stack([np.asarray(c, np.float32) for c in candidates])
    middle = np.median(stack, axis=0)
    spread = [float(np.abs(s[region] - middle[region]).mean()) for s in stack]
    pick = int(np.argmin(spread))
    return candidates[pick], pick, round(max(spread) - min(spread), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path,
                        default=HERE / "records/items_p62_flash38.json")
    parser.add_argument("--tag", default="fx_h10_sub")
    parser.add_argument("--core", default=None)
    parser.add_argument("--halo", type=int, default=HALO)
    parser.add_argument("--draws", type=int, default=1,
                        help="repaint this many times and keep the least idiosyncratic")
    parser.add_argument("--strength", type=float, default=None,
                        help="how far from the photograph the repaint may wander; "
                             "1.0 is pure noise and the pipeline default")
    parser.add_argument("--carrier-whole", action="store_true",
                        help="take the whole box wherever a carrier is declared")
    parser.add_argument("--flood", action="store_true",
                        help="grow the mask along the carrier's own colour until the "
                             "carrier ends, instead of dilating everywhere by a fixed halo")
    parser.add_argument("--flood-tol", type=float, default=14.0)
    parser.add_argument("--flood-grow", type=int, default=30)
    parser.add_argument("--steps", type=int, default=None,
                        help="denoising steps; the pipeline default is 30 and has never "
                             "been varied")
    parser.add_argument("--records", type=Path, default=None,
                        help="where the per-photo rows go; default records/<tag>.jsonl. Give each "
                             "parallel shard its own file, or their lines interleave")
    parser.add_argument("--clip", action="store_true",
                        help="confine the core to the localizer's own boxes before the halo")
    args = parser.parse_args()
    if args.core:
        globals()["CORE"] = HERE / "masks" / args.core
    if args.strength is not None:
        C.STRENGTH = args.strength
    if args.steps is not None:
        C.STEPS = args.steps
    items = {str(i["id"]): i for i in json.loads(args.items.read_text())}

    from pipeline import load_all
    models = load_all()
    out_dir, mask_dir = HERE / "results" / args.tag, HERE / "masks" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = args.records or HERE / f"records/{args.tag}.jsonl"

    with records.open("a", encoding="utf-8", buffering=1) as handle:
        for n, image_id in enumerate(sorted(items), 1):
            if (out_dir / f"{image_id}.png").exists():
                continue
            core_path = CORE / f"{image_id}.png"
            if not core_path.exists():
                continue
            item = items[image_id]
            image = Image.open(item["source"]).convert("RGB")
            core = np.asarray(Image.open(core_path).convert("L")) > 127
            started = time.perf_counter()
            if not core.any():
                image.save(out_dir / f"{image_id}.png")
                handle.write(json.dumps({"id": image_id, "arm": args.tag,
                                         "changed": False}) + "\n")
                continue
            was = int(core.sum())
            marked = boxes_of(item, image.size, False)
            escaped = 0
            if args.clip:
                # masking.build's own docstring says the two signals are "unioned, then
                # confined to the boxes", and only one of them is: the change region is
                # intersected with the boxes, the SAM2 outline never was. SAM2 answers
                # "what object is here", so on a watermark sitting on a car it can return
                # the car -- and on 9977001 it returned enough of one to take the licence
                # plate with it, 28.6% of that mask lying outside every box the localizer
                # drew. Confining the core is what the docstring always claimed.
                inside = np.zeros(core.shape, bool)
                for left, top, right, bottom in marked:
                    inside[top:bottom, left:right] = True
                escaped = int((core & ~inside).sum())
                core = core & inside
            filled_boxes = 0
            if args.carrier_whole:
                # The prompt requires box_2d to contain the whole carrier, out to its own
                # outer edge. So on a region that declares one, every pixel of the box is
                # part of the added layer and the evidence mask has no shrinking left to do
                # -- measured, it covers 91.3% of such a box and leaves the rest standing,
                # which is exactly the leftover rim a uniform halo keeps failing to reach.
                # Where there is no carrier the evidence still earns its keep: it fills only
                # 72.8% of those boxes, and the other 27% is photograph.
                size = image.size
                for region in item["boxes"]:
                    if region.get("physically_in_scene"):
                        continue
                    if not region.get("carrier_present"):
                        continue
                    left, top, right, bottom = masking.widen(
                        masking.to_pixels(region["box_2d"], size), size)
                    if right > left and bottom > top:
                        core[max(0, top):bottom, max(0, left):right] = True
                        filled_boxes += 1
            cap = max(40, int(BRIDGE * max(core.shape)))
            core = close_rows(core, marked, cap)
            flooded = 0
            if args.flood:
                # A uniform halo pays everywhere for a rim that exists in only some places:
                # at 15px it buys carrier coverage and costs a fifth of the photograph, at 0
                # it keeps the photograph and leaves the rim. The carrier's own colour is
                # what separates those cases, so follow it outward one ring at a time, per
                # component, and keep only growth that *terminates* -- a panel has an edge
                # and a wall does not.
                core, flooded = flood(image, core, args.flood_tol, args.flood_grow)
            grown = cv2.dilate((core * 255).astype(np.uint8), rect(args.halo))
            guard = (guard_mask(item, image.size) * 255).astype(np.uint8)
            grown = np.where(guard > 127, 0, grown).astype(np.uint8)
            if not (grown > 127).any():
                image.save(out_dir / f"{image_id}.png")
                handle.write(json.dumps({"id": image_id, "arm": args.tag,
                                         "changed": False}) + "\n")
                continue
            mask = Image.fromarray(grown, "L")
            mask.save(mask_dir / f"{image_id}.png")
            result, pick, gap = draw_best(models, image, mask, seed_for(image_id),
                                          args.draws, grown > 127)
            result, restored = keep_protected(image, result, guard > 127)
            result.save(out_dir / f"{image_id}.png")
            row = {"id": image_id, "arm": args.tag, "changed": True,
                   "core_px": was, "escaped_px": escaped, "draw": pick, "draw_gap": gap,
                   "carrier_boxes_filled": filled_boxes, "flooded_px": flooded,
                   "closed_px": int(core.sum()) - was + escaped,
                   "protected_px_restored": restored,
                   "mask_pct": round(100 * float((grown > 127).mean()), 2),
                   "seconds": round(time.perf_counter() - started, 2)}
            handle.write(json.dumps(row) + "\n")
            print(json.dumps({"n": n, "of": len(items), **row}), flush=True)
    print(json.dumps({"finished": args.tag}), flush=True)


if __name__ == "__main__":
    main()
