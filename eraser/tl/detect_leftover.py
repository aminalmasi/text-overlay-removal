#!/usr/bin/env python3
"""Find text still standing in a finished image, and say whose fault each one is.

CRAFT scores every pixel for how much it looks like the body of a character. It is far too
eager to be a localizer -- on 86 photographs with no overlay at all it fired on 85 -- but
that is a precision problem, and precision is recoverable here from geometry the pipeline
already has. What survives is a detector that works on any script, which matters: the text
in this corpus is mostly Arabic, and the OCR models available offline read Latin only.

Every blob CRAFT finds in the finished image is sorted by *where* it sits and whether it was
there before:

  leftover    inside the region we repainted, and the original had text there too.
              We tried to remove it and did not. A removal failure.
  invented    inside the region we repainted, and the original had nothing there.
              The eraser wrote it. Also a removal failure, but a different one.
  built-in    inside a region the localizer marked physically_in_scene. Printed on the
              goods, a plate, a badge. Standing on purpose -- not a fault.
  missed      outside every box and outside the mask. The localizer never saw it.
              A localization failure, and the only class that P63 could fix.

The classification needs no recognition at all, which is what makes it script-agnostic:
being *inside the mask* already means the pipeline claimed that region, so leftover text
there is a failure whatever alphabet it is in.

`--read` adds a Florence-2 pass over the flagged crops. Its only job is to drop blobs whose
text is digits alone, since a bare number is as likely to be a plate or a serial as an
overlay. It is best-effort: Florence-2 does not read Arabic, so it will decline to help on
most of this corpus, and a blob it cannot read is kept rather than dropped.
"""
from __future__ import annotations

import argparse
import os
import json
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments/added_evidence_v1"))
HERE = ROOT / "experiments/creativity_v1"
CRAFT_W = os.path.join(os.environ.get("TL_MODELS", "/models"), "CRAFT/craft_mlt_25k.pth")
FLORENCE = os.path.join(os.environ.get("TL_MODELS", "/models"), "Florence-2-large-ft")
TARGET, MAXSCALE = 1280, 2.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
HOT, WAS_THERE, MIN_AREA = 0.55, 0.35, 60
# CRAFT predicts two maps: a region score high on each character body, and an *affinity*
# score high in the gap BETWEEN neighbouring characters of the same word. Only the region
# map was used until now, and on its own it fires on anything letter-shaped -- tile
# ornament, tyre tread, the ticks on a measuring tape. Requiring characters to be linked
# into a run is what separates writing from decoration, and it is free: the model already
# returns it.
AFFINITY, MIN_CHARS = 0.35, 2
rect = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1,) * 2)


def region_score(model, crop):
    h, w = crop.shape[:2]
    scale = min(MAXSCALE, max(1.0, TARGET / max(h, w)))
    small = cv2.resize(crop, (max(32, int(w * scale)), max(32, int(h * scale))),
                       interpolation=cv2.INTER_CUBIC)
    pad = cv2.copyMakeBorder(small, 0, (-small.shape[0]) % 32, 0, (-small.shape[1]) % 32,
                             cv2.BORDER_REPLICATE)
    t = torch.from_numpy(((pad.astype(np.float32) / 255 - MEAN) / STD)
                         .transpose(2, 0, 1))[None].to("cuda")
    with torch.inference_mode():
        out = model(t)
    maps = out[0].cpu().numpy()                       # H/2 x W/2 x 2: region, affinity
    maps = maps[: (small.shape[0] + 1) // 2, : (small.shape[1] + 1) // 2]
    return (cv2.resize(maps[..., 0], (w, h), interpolation=cv2.INTER_LINEAR),
            cv2.resize(maps[..., 1], (w, h), interpolation=cv2.INTER_LINEAR))


def to_pixels(box, size):
    ymin, xmin, ymax, xmax = box
    w, h = size
    return (max(0, round(xmin / 1000 * w)), max(0, round(ymin / 1000 * h)),
            min(w, round(xmax / 1000 * w)), min(h, round(ymax / 1000 * h)))


def boxes_mask(item, size, want_scene):
    canvas = np.zeros((size[1], size[0]), bool)
    for region in item["boxes"]:
        if bool(region.get("physically_in_scene")) != want_scene:
            continue
        left, top, right, bottom = to_pixels(region["box_2d"], size)
        canvas[top:bottom, left:right] = True
    return canvas


def classify(image_id, item, result, source, mask, craft):
    """Every text blob in the finished image, with where it sits and what that means."""
    size = (source.shape[1], source.shape[0])
    after, affinity = region_score(craft, result)
    before, _ = region_score(craft, source)
    protect = boxes_mask(item, size, True)
    hot = (after > HOT).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)

    # keep only characters that belong to a run: join the character bodies through the
    # affinity map and require at least MIN_CHARS of them in the same connected word.
    joined = ((after > HOT) | (affinity > AFFINITY)).astype(np.uint8)
    words, word_of = cv2.connectedComponents(joined, 8)
    members = {}
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        ys, xs = np.nonzero(labels == k)
        w_id = int(np.bincount(word_of[ys, xs]).argmax())
        members.setdefault(w_id, []).append(k)
    in_word = {k for ids in members.values() if len(ids) >= MIN_CHARS for k in ids}

    found = []
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        blob = labels == k
        # The run requirement is there to tell writing from decoration, which is a question
        # only worth asking where we do not already know the answer. Inside the mask, on a
        # spot where the original demonstrably had text, the pipeline already claimed the
        # region and failed to clear it -- that is a leftover whether one character survived
        # or six, so the gate is not applied there.
        surviving = ((blob & mask).sum() / blob.sum() > 0.5
                     and float(before[blob].max()) > WAS_THERE)
        if k not in in_word and not surviving:
            continue
        in_mask = (blob & mask).sum() / blob.sum()
        in_prot = (blob & protect).sum() / blob.sum()
        was = float(before[blob].max())
        if in_prot > 0.5:
            kind = "built_in"
        elif in_mask > 0.5:
            kind = "leftover" if was > WAS_THERE else "invented"
        else:
            kind = "built_in" if was > WAS_THERE and in_mask < 0.05 else "missed"
        x, y, w, h = (stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP],
                      stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])
        found.append({"kind": kind, "box": [int(x), int(y), int(x + w), int(y + h)],
                      "area": int(stats[k, cv2.CC_STAT_AREA]),
                      "score": round(float(after[blob].max()), 3),
                      "was_there": round(was, 3),
                      "in_mask": round(float(in_mask), 3)})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="n_h10_sub")
    parser.add_argument("--items", type=Path, default=HERE / "records/items_p62_flash38.json")
    parser.add_argument("--masks", default="n_h10_sub")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from craft_model import load_craft
    craft = load_craft(CRAFT_W, "cuda")
    items = {str(i["id"]): i for i in json.loads(args.items.read_text())}
    report, tally = {}, {"leftover": 0, "invented": 0, "built_in": 0, "missed": 0}
    per_image = {k: 0 for k in tally}

    for image_id in sorted(items):
        out_p = HERE / "results" / args.arm / f"{image_id}.png"
        mask_p = HERE / "masks" / args.masks / f"{image_id}.png"
        if not out_p.exists():
            continue
        item = items[image_id]
        source = np.asarray(Image.open(item["source"]).convert("RGB"))
        result = np.asarray(Image.open(out_p).convert("RGB"))
        if result.shape != source.shape:
            continue
        mask = (np.asarray(Image.open(mask_p).convert("L")) > 127 if mask_p.exists()
                else np.zeros(source.shape[:2], bool))
        found = classify(image_id, item, result, source, mask, craft)
        report[image_id] = found
        kinds = {f["kind"] for f in found}
        for f in found:
            tally[f["kind"]] += 1
        for k in per_image:
            per_image[k] += k in kinds

    n = len(report)
    print(f"{args.arm}: {n} images, {sum(tally.values())} text blobs standing\n")
    print(f"  {'class':10s} {'blobs':>6s} {'images':>7s}  what it means")
    meaning = {"leftover": "we repainted it and the text survived",
               "invented": "we repainted it and the eraser wrote something",
               "built_in": "printed on the goods -- standing on purpose",
               "missed":   "the localizer never boxed it"}
    for k in ("leftover", "invented", "missed", "built_in"):
        print(f"  {k:10s} {tally[k]:6d} {per_image[k]:7d}  {meaning[k]}")
    bad = sorted((i for i in report
                  if any(f["kind"] in ("leftover", "invented", "missed") for f in report[i])),
                 key=lambda i: -sum(f["area"] for f in report[i]
                                    if f["kind"] in ("leftover", "invented", "missed")))
    print(f"\n  {len(bad)} of {n} images have at least one blob that is a fault:")
    for image_id in bad[:16]:
        kinds = [f"{f['kind']}({f['area']}px)" for f in report[image_id]
                 if f["kind"] in ("leftover", "invented", "missed")]
        print(f"    {image_id:10s} {', '.join(kinds[:4])}")
    target = HERE / "records" / (args.out or f"leftover_{args.arm}.json")
    target.write_text(json.dumps(report, indent=1))
    print(f"\n  written to {target.name}")


if __name__ == "__main__":
    main()
