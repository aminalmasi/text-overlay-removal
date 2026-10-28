#!/usr/bin/env python3
"""Refine Gemini's erase boxes from evidence in the photograph, before the pipeline sees them.

Two systematic box errors in the compare_steps review (2026-10-10), neither one a prompt can fix
reliably on flash-lite:
  - boxes cut through words: 7-12% of erase regions hold only part of a CRAFT word, and the part left
    outside is mostly above or below (Arabic dots, ascenders and descenders). The pipeline widens a box
    by 6% of its height, about 2 px on one line, so those strokes are never in the mask.
  - emoji at line ends are left out: Gemini marks the region kind "mixed" (text and emoji) but fits
    box_2d to the letters. CRAFT does not respond to emoji, so only the region's own label says so.

Rule 1, snap: a CRAFT word (region score > 0.25, letters joined along the row) at least a fifth inside
the widened box pulls the box out to its full extent -- by at most half a word height up or down and
one word height sideways, so a long physical text line crossing the box cannot drag it along.
Rule 2, mixed: a region of kind "mixed" with no carrier is extended left and right by one word height.
Rule 0, align (--align): Gemini sometimes places a whole reply's boxes off their text (s042 P91: every
line half a line low; s191 P91: a block 11% of the height too low). For a plain-text region with no
carrier, slide its box_2d within +-50% of its width and +-60% of its height to where it holds the
most CRAFT text. If that holds at least 1.3x the text and moves at least 15% of the box: when the box
as drawn holds under 30% of it, move the box there; otherwise stretch it to cover both positions.
Logos, emoji and carriers are never moved -- CRAFT cannot see them, so nearby text would pull them off.
Protected regions are never changed. Everything else in the item is copied as it is.

    python3 refine_boxes.py --items records/items_X91.json --tag X91R
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # deploy copy
from pipeline import masking                        # noqa: E402

LEVEL, MOST = 0.25, 0.2


def to_1000(box, size):
    l, t, r, b = box
    w, h = size
    return [round(1000 * t / h), round(1000 * l / w), round(1000 * b / h), round(1000 * r / w)]


def box_mass(I, l, t, r, b, shape):
    h, w = shape
    l, r = max(0, min(w, l)), max(0, min(w, r))
    t, b = max(0, min(h, t)), max(0, min(h, b))
    return 0 if r <= l or b <= t else int(I[b, r] - I[t, r] - I[b, l] + I[t, l])


def align(region, I, size):
    """The move or stretch of rule 0, as a new box_2d in 0..1000, or None."""
    if region.get("carrier_present") or region.get("kind", "text") != "text":
        return None, None
    tb = region["box_2d"]                # with no carrier box_2d IS the text box; text_box_2d can be
    if not (isinstance(tb, list) and len(tb) == 4) or tb[0] >= tb[2] or tb[1] >= tb[3]:   # garbage (s191)
        return None, None
    w, h = size
    l, t, r, b = masking.to_pixels(tb, size)
    bw, bh = r - l, b - t
    if bw < 6 or bh < 6:
        return None, None
    m0 = box_mass(I, l, t, r, b, (h, w))
    if m0 < 0.03 * bw * bh and m0 < 40:
        return None, None
    best = (m0, 0, 0)
    for dy in range(-int(0.6 * bh), int(0.6 * bh) + 1, max(2, bh // 25)):
        for dx in range(-int(0.5 * bw), int(0.5 * bw) + 1, max(2, bw // 25)):
            m = box_mass(I, l + dx, t + dy, r + dx, b + dy, (h, w))
            if m > best[0]:
                best = (m, dx, dy)
    mb, dx, dy = best
    if not (mb >= 1.3 * max(m0, 1) and (abs(dx) >= 0.15 * bw or abs(dy) >= 0.15 * bh)):
        return None, None
    bl, bt, br, bb = masking.to_pixels(region["box_2d"], size)
    moved = (bl + dx, bt + dy, br + dx, bb + dy)
    if m0 < 0.3 * mb:
        new, why = moved, "align-move"
    else:
        new, why = (min(bl, moved[0]), min(bt, moved[1]), max(br, moved[2]), max(bb, moved[3])), "align-stretch"
    new = (max(0, new[0]), max(0, new[1]), min(w, new[2]), min(h, new[3]))
    return to_1000(new, size), why


def refine(region, words, labels, stats, size, mixed_rule=True, I=None):
    why0 = []
    if I is not None:
        moved, how = align(region, I, size)
        if moved is not None:
            region = {**region, "box_2d": moved, "box_2d_gemini": region["box_2d"]}
            why0 = [how]
    l, t, r, b = masking.to_pixels(region["box_2d"], size)
    wl, wt, wr, wb = masking.widen((l, t, r, b), size)
    if wr <= wl or wb <= wt:
        return region, []
    inside = labels[wt:wb, wl:wr]
    ids, counts = np.unique(inside[inside > 0], return_counts=True)
    held = [k for k, c in zip(ids, counts) if c >= MOST * stats[k, cv2.CC_STAT_AREA]]
    heights = [stats[k, cv2.CC_STAT_HEIGHT] for k in held]
    line = float(np.median(heights)) if heights else float(b - t)
    nl, nt, nr, nb = l, t, r, b
    for k in held:                      # only text that escapes the widened box moves an edge
        x, y, w, h = stats[k, :4]
        if x < wl:
            nl = min(nl, max(x, l - line))
        if x + w > wr:
            nr = max(nr, min(x + w, r + line))
        if y < wt:
            nt = min(nt, max(y, t - line / 2))
        if y + h > wb:
            nb = max(nb, min(y + h, b + line / 2))
    why = list(why0)
    if (nl, nt, nr, nb) != (l, t, r, b):
        why.append("snap")
    if mixed_rule and region.get("kind") == "mixed" and not region.get("carrier_present"):
        nl, nr = nl - line, nr + line
        why.append("mixed")
    nl, nt = max(0, int(round(nl))), max(0, int(round(nt)))
    nr, nb = min(size[0], int(round(nr))), min(size[1], int(round(nb)))
    if not why:
        return region, []
    return {**region, "box_2d": to_1000((nl, nt, nr, nb), size),
            "box_2d_gemini": region.get("box_2d_gemini", region["box_2d"])}, why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--craft", default="craft_full", help="masks/<craft>: whole-photo CRAFT scores")
    ap.add_argument("--no-mixed", action="store_true")
    ap.add_argument("--align", action="store_true", help="rule 0: move or stretch plain-text boxes Gemini placed off their text")
    ap.add_argument("--shards", type=int, default=2)
    a = ap.parse_args()
    out, changed, grew = [], [], []
    for item in json.loads(a.items.read_text(encoding="utf-8")):
        score = np.asarray(Image.open(HERE / "masks" / a.craft / f"{item['id']}.png")) / 255.0
        h, w = score.shape
        words = cv2.dilate((score > LEVEL).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 160), 3)))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(words, 8)
        I = cv2.integral(cv2.dilate((score > LEVEL).astype(np.uint8), np.ones((3, 3), np.uint8))) if a.align else None
        boxes, reasons = [], []
        for region in item["boxes"]:
            if not isinstance(region, dict) or region.get("physically_in_scene") is not False \
                    or not isinstance(region.get("box_2d"), list) or len(region["box_2d"]) != 4:
                boxes.append(region)
                continue
            new, why = refine(region, words, labels, stats, (w, h), not a.no_mixed, I)
            boxes.append(new)
            reasons += why
            if why:
                o, n = masking.to_pixels(region["box_2d"], (w, h)), masking.to_pixels(new["box_2d"], (w, h))
                grew.append(((n[2] - n[0]) * (n[3] - n[1]) - (o[2] - o[0]) * (o[3] - o[1])) / (w * h))
        if reasons:
            changed.append((item["id"], sorted(set(reasons))))
        out.append({**item, "boxes": boxes, "refined": sorted(set(reasons))})
    (HERE / f"records/items_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    only = [i for i in out if i["refined"]]
    (HERE / f"records/items_{a.tag}_changed.json").write_text(json.dumps(only, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in range(a.shards):
        (HERE / f"records/items_{a.tag}_changed_shard_{n}.json").write_text(
            json.dumps(only[n::a.shards], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tag": a.tag, "photos": len(out), "photos_changed": len(changed),
                      "by_rule": {r: sum(r in why for _, why in changed) for r in ("align-move", "align-stretch", "snap", "mixed")},
                      "box_area_added_pct_of_frame_median": round(100 * float(np.median(grew)), 2) if grew else 0,
                      "changed": [i for i, _ in changed]}))


if __name__ == "__main__":
    main()
