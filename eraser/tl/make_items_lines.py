#!/usr/bin/env python3
"""The erasure pipeline's input for one worker arm of the s4lines run (P88, P91, P92 on flash-lite).

Same selection rule as make_items_arm.py: the gate says removal 1 AND the worker reports at least
one region with physically_in_scene false. The gate is not in s4lines -- it gave the same answer on
218 of 219 photos in s4worker -- so its s4worker answers are used, matched by the photo's hash.

--split-lines (P92 only) is the point of P92: a region with no carrier is replaced by one region
per line box, each with box_2d = text_box_2d = that line, so the pipeline erases the lines and
keeps the photograph between them. A region with a carrier keeps its box_2d, carrier and all. A
region whose line boxes are missing or all malformed is kept as Gemini drew it.

    python3 make_items_lines.py --worker p92.lite --tag X92L --split-lines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def valid(box):
    return (isinstance(box, list) and len(box) == 4 and all(type(x) is int for x in box)
            and all(0 <= x <= 1000 for x in box) and box[0] < box[2] and box[1] < box[3])


def split_lines(boxes):
    """Bare regions become one region per line box; everything else is left as it is."""
    out, split = [], 0
    for region in boxes:
        lines = []
        if isinstance(region, dict):              # P92/P94 line_boxes, or P93 lines [{text, box_2d}]
            raw = region.get("line_boxes") or [l.get("box_2d") for l in region.get("lines") or [] if isinstance(l, dict)]
            lines = [b for b in raw if valid(b)]
        if (not isinstance(region, dict) or region.get("physically_in_scene") is not False
                or region.get("carrier_present") or not lines):
            out.append(region)
            continue
        split += 1
        for n, line in enumerate(lines):
            out.append({**{k: v for k, v in region.items() if k not in ("line_boxes", "lines")},
                        "box_2d": line, "text_box_2d": line, "split_from_line": n})
    return out, split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "returns/s4lines/unpacked/results")
    ap.add_argument("--gate-results", type=Path, default=HERE / "returns/s4worker/unpacked/results")
    ap.add_argument("--gate", default="p84.lite")
    ap.add_argument("--worker", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split-lines", action="store_true")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--key", default="records/s4lines_key_PRIVATE.json", help="the pack's private id key")
    ap.add_argument("--from-label", default="s4lines", help="where these boxes came from, for the items")
    a = ap.parse_args()

    key = json.loads((HERE / a.key).read_text())
    gate_key = json.loads((HERE / "records/s4worker_key_PRIVATE.json").read_text())
    k65 = json.loads((HERE / "records/p65pair_key_PRIVATE.json").read_text())
    sid = {v["sha1"]: s for s, v in k65.items()}
    load = lambda d, n: {e["id"]: e for e in json.loads((d / f"{n}.json").read_text(encoding="utf-8"))["images"]}
    gate_by_sha = {gate_key[w]["sha1"]: e for w, e in load(a.gate_results, a.gate).items()}
    worker = load(a.results, a.worker)
    items, no_reply, regions_split = [], [], 0
    for x in sorted(key):
        g = gate_by_sha[key[x]["sha1"]].get("reply") or {}
        if g.get("removal") != 1:
            continue
        if "reply" not in worker[x]:
            no_reply.append(x)                   # cut off or failed: production would erase nothing
            continue
        boxes = worker[x]["reply"].get("boxes") or []
        if not any(isinstance(b, dict) and b.get("physically_in_scene") is False for b in boxes):
            continue
        if a.split_lines:
            boxes, split = split_lines(boxes)
            regions_split += split
        items.append({"id": sid[key[x]["sha1"]], "source": key[x]["source"],
                      "from": f"{a.worker}/{a.from_label}" + ("/split" if a.split_lines else ""),
                      "s4lines_id": x, "boxes": boxes})
    items.sort(key=lambda i: i["id"])
    (HERE / f"records/items_{a.tag}.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    for n in range(a.shards):
        (HERE / f"records/items_{a.tag}_shard_{n}.json").write_text(
            json.dumps(items[n::a.shards], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"tag": a.tag, "worker": a.worker, "erased": len(items),
                      "regions_split_into_lines": regions_split,
                      "gate_said_1_but_no_worker_reply": no_reply}))


if __name__ == "__main__":
    main()
